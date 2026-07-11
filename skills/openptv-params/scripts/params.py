"""openptv-params — inspect and mutate OpenPTV YAML parameter files without the GUI."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _load(yaml_path: Path) -> dict:
    import yaml
    return yaml.safe_load(yaml_path.read_text())


def _save(data: dict, yaml_path: Path) -> None:
    import yaml
    yaml_path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))


def _backup(yaml_path: Path) -> Path:
    bak = yaml_path.with_suffix(".parbak")
    shutil.copy2(yaml_path, bak)
    return bak


def _find_yaml(dataset: Path) -> Path | None:
    for pat in ("parameters_*.yaml", "*.yaml"):
        hits = sorted(dataset.glob(pat))
        if hits:
            return hits[0]
    return None


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_show(args) -> int:
    import yaml
    base = Path(args.dataset).resolve()
    yp = _find_yaml(base)
    if yp is None:
        print("ERROR: no YAML found", file=sys.stderr)
        return 1
    data = _load(yp)
    if args.section:
        data = data.get(args.section)
        if data is None:
            print(f"ERROR: section '{args.section}' not found", file=sys.stderr)
            return 1
    print(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), end="")
    return 0


def cmd_set(args) -> int:
    base = Path(args.dataset).resolve()
    yp = _find_yaml(base)
    if yp is None:
        print("ERROR: no YAML found", file=sys.stderr)
        return 1

    parts = args.key.split(".", 1)
    if len(parts) != 2:
        print("ERROR: key must be section.field (e.g. ptv.mmp_n2)", file=sys.stderr)
        return 1
    section, field = parts

    data = _load(yp)
    if section not in data:
        print(f"ERROR: section '{section}' not found", file=sys.stderr)
        return 1

    # coerce value
    raw = args.value
    try:
        val = int(raw)
    except ValueError:
        try:
            val = float(raw)
        except ValueError:
            if raw.lower() in ("true", "false"):
                val = raw.lower() == "true"
            else:
                val = raw

    old = data[section].get(field, "<missing>")
    data[section][field] = val

    bak = _backup(yp)
    _save(data, yp)
    print(f"Set {section}.{field}: {old!r} → {val!r}  (backup: {bak.name})")
    return 0


def cmd_validate(args) -> int:
    base = Path(args.dataset).resolve()
    yp = _find_yaml(base)
    if yp is None:
        print("ERROR: no YAML found", file=sys.stderr)
        return 1

    data = _load(yp)
    errors: list[str] = []
    warnings: list[str] = []

    # refractive indices
    ptv = data.get("ptv", {})
    n1, n2, n3 = ptv.get("mmp_n1"), ptv.get("mmp_n2"), ptv.get("mmp_n3")
    if None in (n1, n2, n3):
        errors.append("ptv.mmp_n1/n2/n3 missing")
    else:
        if abs(float(n1) - 1.0) > 0.05:
            warnings.append(f"ptv.mmp_n1={n1} — expected ~1.0 (air)")
        if float(n2) < float(n3):
            errors.append(
                f"ptv.mmp_n2={n2} < mmp_n3={n3} — looks swapped "
                "(n2=glass~1.46, n3=water~1.33)"
            )

    # image size
    for k in ("imx", "imy"):
        if ptv.get(k, 0) <= 0:
            errors.append(f"ptv.{k} must be > 0")

    # pixel size
    for k in ("pix_x", "pix_y"):
        if ptv.get(k, 0) <= 0:
            errors.append(f"ptv.{k} must be > 0")

    # sequence range
    seq = data.get("sequence", {})
    first, last = seq.get("first"), seq.get("last")
    if first is not None and last is not None and int(first) > int(last):
        errors.append(f"sequence.first={first} > sequence.last={last}")

    # calblock
    fixp = data.get("cal_ori", {}).get("fixp_name")
    if fixp:
        fp = (base / fixp) if not Path(fixp).is_absolute() else Path(fixp)
        if not fp.exists():
            errors.append(f"calblock not found: {fp}")
    else:
        warnings.append("cal_ori.fixp_name missing")

    # tracking params sanity
    track = data.get("track", {})
    for ax in ("dvxmin", "dvymin", "dvzmin"):
        if track.get(ax, -1) >= 0:
            warnings.append(f"track.{ax}={track[ax]} — expected negative")
    for ax in ("dvxmax", "dvymax", "dvzmax"):
        if track.get(ax, 1) <= 0:
            warnings.append(f"track.{ax}={track[ax]} — expected positive")

    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARN:  {w}")
    if not errors and not warnings:
        print("OK")
    return 1 if errors else 0


def cmd_diff(args) -> int:
    import yaml
    p1, p2 = Path(args.yaml1), Path(args.yaml2)
    d1 = _load(p1)
    d2 = _load(p2)

    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key + "."))
            else:
                out[key] = v
        return out

    f1, f2 = _flatten(d1), _flatten(d2)
    all_keys = sorted(set(f1) | set(f2))
    changed = False
    for k in all_keys:
        v1, v2 = f1.get(k, "<missing>"), f2.get(k, "<missing>")
        if v1 != v2:
            print(f"- {k}: {v1!r}")
            print(f"+ {k}: {v2!r}")
            changed = True
    if not changed:
        print("(no differences)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="openptv parameter inspector/editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show", help="print parameters (or one section) as YAML")
    p.add_argument("dataset")
    p.add_argument("--section", default=None, help="restrict to one section (e.g. ptv, track)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("set", help="set one parameter field and rewrite YAML")
    p.add_argument("dataset")
    p.add_argument("key", help="section.field (e.g. ptv.mmp_n2)")
    p.add_argument("value", help="new value (auto-coerced to int/float/bool/str)")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("validate", help="sanity-check parameters")
    p.add_argument("dataset")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("diff", help="diff two parameter YAML files")
    p.add_argument("yaml1")
    p.add_argument("yaml2")
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
