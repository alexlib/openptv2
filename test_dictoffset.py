class RegularClass:
    pass

from dataclasses import dataclass


@dataclass
class DataClass:
    x: int

class SlottedClass:
    __slots__ = ['x']

print("RegularClass dictoffset:", RegularClass.__dictoffset__)
print("DataClass dictoffset:", DataClass.__dictoffset__)
print("SlottedClass dictoffset:", SlottedClass.__dictoffset__)
