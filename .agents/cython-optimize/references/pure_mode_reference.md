# Cython 3 Pure Python Mode — Syntax Reference

Source: https://cython.readthedocs.io/en/latest/src/tutorial/pure.html
(condensed; consult the live page for anything not covered here)

Every example below shows the Pure Python form on top and the equivalent
`.pyx` Cython form below it, so you can translate in either direction.

## Table of contents
- [Compiled switch](#compiled-switch)
- [Static typing (declare, locals, returns, exceptval)](#static-typing)
- [C types](#c-types)
- [Extension types and cdef functions](#extension-types-and-cdef-functions)
- [GIL control](#gil-control)
- [cimports](#cimports)
- [Structs, unions, typedef, cast, fused types, address, sizeof, typeof](#further-declarations)
- [Augmenting with a .pxd file](#augmenting-pxd)
- [PEP-484 / PEP-526 annotations](#pep-484-annotations)
- [typing module support](#typing-module)
- [Reference table: how annotations are interpreted](#reference-table)
- [Tips: avoiding the cython runtime dependency, calling C functions, C arrays](#tips)

## Compiled switch

`cython.compiled` is `True` only inside the compiled extension, `False` under
the plain interpreter:

```python
if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")
```

## Static typing

`cython.declare(name=type[, value])` — declares a typed variable (like
`cdef type var = value`). Two forms:

```python
x = cython.declare(cython.int)  # cdef int x
y = cython.declare(cython.double, 0.57721)  # cdef double y = 0.57721
cython.declare(
    x=cython.int, y=cython.double
)  # cdef int x; cdef double y (function-call form)
```

Inside a class body / `@cython.cclass`, `declare` sets attribute visibility:

```python
@cython.cclass
class A:
    cython.declare(a=cython.int, b=cython.int)  # private (default)
    c = cython.declare(cython.int, visibility="public")
    e = cython.declare(cython.int, visibility="readonly")
```

`@cython.locals(name=type, ...)` — types local variables and/or arguments:

```python
@cython.locals(a=cython.long, b=cython.long, n=cython.longlong)
def foo(a, b, x, y):
    n = a * b
```

`@cython.returns(<type>)` — declares the function's return type.

`@cython.exceptval(value=None, *, check=False)` — declares the C-level
exception-signalling value and whether to also check the Python error
indicator. With type annotations and no explicit `@exceptval`, the default
for a numeric return type is `@exceptval(-1, check=True)` (safe, but adds a
check on every call). Use `@exceptval(check=False)` on hot leaf functions
that can't actually raise, to skip the check — but any exception the
function does raise will then only be printed and swallowed, not propagated.

## C types

Built-in: `char, short, int, long, longlong` and unsigned variants
(`uchar, ushort, uint, ulong, ulonglong`); `bint` for C booleans;
`Py_ssize_t` for container sizes. Pointer types: `p_int`, `pp_int`, etc.
(three levels deep interpreted, unlimited compiled), or build one with
`cython.pointer(cython.int)`. Arrays: `cython.int[10]`.
Qualifiers: `cython.const[T]`, `cython.volatile[T]`.

Annotation aliases: Python `bool` → C `bint`, Python `float` → C `double`.
Python `int` has **no** C equivalent and is NOT aliased — it stays a full
Python int object unless you write `cython.int` explicitly. This is the
single most common surprise when porting existing type-hinted code.

`list`, `dict`, `tuple`, and user-defined types can be used as annotations
too (see the reference table below for what they mean when compiled).

## Extension types and cdef functions

- `@cython.cclass` → `cdef class` (extension type, faster attribute access,
  no arbitrary attribute dict unless one is declared).
- `@cython.cfunc` → `cdef` function: C-level call convention, not visible to
  Python-level code once compiled (but the plain-Python shadow module makes
  it work interpreted, so tests still run against the .py file directly).
- `@cython.ccall` → `cpdef` function: callable at the C level *and* from
  Python.
- `@cython.locals(...)` also covers argument types for the above.
- `@cython.inline` → C `inline`.
- `@cython.final` → prevents subclassing / method override; enables
  devirtualization of method calls.

```python
@cython.cfunc
def c_compare(a: cython.int, b: cython.int) -> cython.bint:
    return a == b
```

## GIL control

```python
with cython.nogil:
    pass  # release the GIL for this block


@cython.nogil
@cython.cfunc
@cython.returns(cython.int)
def func_not_needing_the_gil() -> cython.int:
    return 1


with cython.gil:
    pass  # (re)acquire the GIL for this block
```

Note the asymmetry: the context manager form of `nogil` *releases* the GIL
for that block; the decorator form only *marks* a function as callable
without holding the GIL (needed for e.g. calling it from other nogil code or
`prange`). Both accept a compile-time-constant boolean,
e.g. `with cython.nogil(condition):`, useful with fused types where GIL
handling differs per specialization. `@cython.with_gil` is not supported —
use the `with cython.gil:` block instead.

## cimports

`from cython.cimports.<pkg> import <name>` gives access to compile-time
cimports from Python-syntax code (this does NOT make C libraries callable
from plain Python — running the unmodified file still fails on that line if
it's actually invoked at interpreted time unless you guard it):

```python
from cython.cimports.libc import math


def use_libc_math():
    return math.ceil(5.5)
```

## Further declarations

```python
# address-of, equivalent to &x
cython.declare(x=cython.int, x_ptr=cython.p_int)
x_ptr = cython.address(x)

# sizeof
print(cython.sizeof(cython.longlong))

# typeof (debugging)
print(cython.typeof(n))

# struct / union
MyStruct = cython.struct(x=cython.int, y=cython.int, data=cython.double)
a: MyStruct

# typedef
T = cython.typedef(cython.p_int)
cython.declare(my_ptr=T)

# cast — cython.cast(T, t) == <T>t ; typecheck=True == <T?>t
l1 = cython.cast(list, obj)
l2 = cython.cast(list, obj, typecheck=True)

# fused types (compile-time generics over a small type set)
my_fused_type = cython.fused_type(cython.int, cython.float)
```

## Augmenting with a .pxd file

An augmenting `.pxd` with the same base name as the `.py` file overrides
untyped definitions with typed ones, leaving the `.py` file itself untouched.
Tradeoff: you now maintain two files in sync. In the `.pxd`:

- Python-visible functions → `cpdef` (default args become `=*`)
- internal-only functions → `cdef`
- extension types → `cdef class`, with `cdef public` / `cdef readonly` /
  plain `cdef` for attribute visibility
- plain `def` functions cannot be re-typed from a `.pxd` (use `cpdef`
  instead, or type them in the `.py` file directly with decorators)

`cython.declare()` and `@cython.locals` also work inside the `.pxd`:

```python
# dostuff.pxd
import cython
@cython.locals(t=cython.int, i=cython.int)
cpdef int dostuff(int n)
```

Prefer decorators/annotations in the `.py` file itself over a `.pxd` unless
there's a specific reason to keep the `.py` file free of Cython-specific
code (e.g. it must also work standalone in an environment without Cython —
see "Avoiding the cython runtime dependency" below).

## PEP-484 annotations

```python
@cython.ccall
def func(foo: dict, bar: cython.int) -> tuple:
    foo["hello world"] = 3 + bar
    return foo, 5
```

PEP 526 variable annotations also work, and Python-type annotations (`int`,
`float`) shadow the Cython meaning for compatibility — i.e. `a: float` is a C
double, but `b: int` stays a full Python int object:

```python
def func():
    x: cython.int
    y: cython.double = 0.57721
    a: float = 0.54321  # C double
    b: int = 5  # Python int object, not a C int
```

Global-variable annotations are currently ignored (would silently move the
variable out of the module dict) — use `cython.declare()` at module scope
instead if you need a typed global.

`@cython.annotation_typing(False)` (decorator, class decorator, or `with`
block) turns off Cython's interpretation of annotations for that scope, in
case the annotations are there for something else (e.g. a different static
checker) and should be left as plain runtime-ignored hints.

## typing module

Cython 3 understands, from the stdlib `typing` module:
`Optional[T]`, `Union[T, None]` / `Union[None, T]`, `T | None` (all →
"T or None"); typed containers like `List[str]` → `list` with item-type
inference; `Tuple[...]` → C tuple where possible, else Python tuple;
`ClassVar[...]` inside a `cdef class` / `@cython.cclass`.

## Reference table

| Annotation | Meaning when compiled (Cython 3.0) |
|---|---|
| `int` | Exact Python `int` object (language_level=3) |
| `float` | C `double` |
| `dict`, `list`, `list[T]`, etc. | Exact type (no subclasses), not `None` |
| Extension type defined in Cython | That type or a subclass, not `None` |
| `cython.int`, `cython.long`, etc. | Equivalent C numeric type |
| `typing.Optional[T]` / `Union[T, None]` / `T \| None` | `T` (must be a Python object type), allows `None` |
| `typing.List[T]` etc. | Exact list, element type inferred on access, not `None` |
| `typing.ClassVar[...]` | Python-object class variable |

## Tips

**Avoiding the cython runtime dependency** — if the `.py` file must run
without Cython installed at all (not even the shadow module), stub it:

```python
try:
    import cython
except ImportError:

    class _fake_cython:
        compiled = False

        def cfunc(self, func):
            return func

        def ccall(self, func):
            return func

        def __getattr__(self, type_name):
            return "object"

    cython = _fake_cython()
```

**Calling C functions from pure mode** — you generally can't call a C
function directly from uncompiled Python. Pattern: declare it `cpdef` in an
augmenting `.pxd`, and conditionally import a Python equivalent when not
compiled:

```python
# mymodule.pxd
cdef extern from "math.h":
    cpdef double sin(double x)
```

```python
# mymodule.py
import cython

if not cython.compiled:
    from math import sin
print(sin(0))
```

**C arrays for fixed-size lists** — a `cython.int[10]`-typed local coerces
to/from a Python list automatically, letting you swap a small fixed-size
Python list for a C array purely by adding a `@cython.locals` type:

```python
@cython.locals(counts=cython.int[10], digit=cython.int)
def count_digits(digits):
    counts = [0] * 10
    for digit in digits:
        counts[digit] += 1
    return counts
```
