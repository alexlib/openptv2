import sys

import pytest

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
    MmLut,
)
from openptv2.algorithms.correspondences import NTupel

# Loop-Internal Objects
from openptv2.algorithms.epi import Candidate, Coord2d

# Config / Parameters
from openptv2.algorithms.parameters import (
    CalibrationPar,
    ControlPar,
    ExaminePar,
    MmNp,
    MultimediaPar,
    MultiPlanesPar,
    OrientPar,
    PftVersionPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
)
from openptv2.algorithms.segmentation import Peak

# Core Loop Data
from openptv2.algorithms.tracking_frame_buf import (
    Corres,
    Frame,
    FrameBuf,
    Pathinfo,
    Target,
)

CLASSES_TO_TEST = [
    # Core
    Target, Pathinfo, Frame, FrameBuf, Corres,
    # Parameters
    TrackPar, SequencePar, VolumePar, ControlPar, TargetPar,
    MultimediaPar, CalibrationPar, MultiPlanesPar, ExaminePar,
    PftVersionPar, OrientPar, MmNp,
    # Calibration
    Calibration, Exterior, Interior, Glass, AddedPar, MmLut,
    # Internal Loop Objects
    Candidate, Coord2d, NTupel, Peak
]

@pytest.mark.parametrize("cls", CLASSES_TO_TEST)
def test_is_compiled_cclass(cls):
    """
    Validates that the given class is compiled as a Cython extension type (@cython.cclass),
    meaning it behaves like a fast C-struct rather than a slow Python dictionary object.
    
    This fulfills the requirement of Phase 6.9 in optimization_plan.md.
    """

    # 1. The class should not have a dynamic instance dictionary.
    # Regular Python classes (and @dataclass) have a __dictoffset__ != 0 (e.g. -1 or >0).
    # Extension types (and slotted classes) have a __dictoffset__ of 0.
    assert cls.__dictoffset__ == 0, f"{cls.__name__} has a dynamic dictionary (__dictoffset__ != 0). Convert it to @cython.cclass."

    # 2. Check that the module it belongs to is a compiled shared object (.so / .pyd)
    module_name = cls.__module__
    module = sys.modules[module_name]

    assert hasattr(module, "__file__"), f"Module {module_name} has no __file__ attribute."
    assert module.__file__.endswith((".so", ".pyd")), f"Module {module_name} is not compiled! __file__: {module.__file__}"

