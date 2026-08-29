import importlib
import sys
import types

try:
    from traits.etsconfig.etsconfig import ETSConfig

    ETSConfig.toolkit = "qt"
except ModuleNotFoundError:
    # Traits is an optional dependency for headless/non-GUI usage.
    pass

# Create a virtual module 'openptv2.gui.pyptv' so old imports do not crash
pyptv_module = types.ModuleType("openptv2.gui.pyptv")
sys.modules["openptv2.gui.pyptv"] = pyptv_module

# Map legacy submodules to their new locations
submodule_mapping = {
    "pyptv_batch": "openptv2.batch.pyptv_batch",
    "pyptv_batch_parallel": "openptv2.batch.pyptv_batch_parallel",
    "pyptv_batch_plugins": "openptv2.batch.pyptv_batch_plugins",
}

# The following modules have been flattened directly into openptv2.gui
gui_submodules = [
    "ptv",
    "ptv_calibration",
    "flowtracks_utils",
    "parameter_manager",
    "experiment",
    "cli",
    "parameter_defaults",
    "parameter_gui",
    "calibration_gui",
    "detection_gui",
    "mask_gui",
    "code_editor",
    "tracking_debug_utils",
    "parameters",
    "plot_3d_positions",
    "plot_3d_trajectories",
]
for sub in gui_submodules:
    submodule_mapping[sub] = f"openptv2.gui.{sub}"


class LazySubmodule(types.ModuleType):
    """Lazy module shim that dynamically delegates to the new location on access."""

    def __init__(self, name, target):
        super().__init__(name)
        self.__target = target

    def __getattr__(self, attr):
        mod = importlib.import_module(self.__target)
        return getattr(mod, attr)

    def __dir__(self):
        try:
            mod = importlib.import_module(self.__target)
            return dir(mod)
        except ImportError:
            return []


# Register all legacy submodules as lazy shims
for old_sub, target in submodule_mapping.items():
    shim_sub = LazySubmodule(f"openptv2.gui.pyptv.{old_sub}", target)
    sys.modules[f"openptv2.gui.pyptv.{old_sub}"] = shim_sub
    setattr(pyptv_module, old_sub, shim_sub)


def __getattr__(name):
    if name == "__version__":
        from .__version__ import __version__

        return __version__
    if name in gui_submodules:
        return importlib.import_module(f"openptv2.gui.{name}")
    if name in submodule_mapping:
        return importlib.import_module(submodule_mapping[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
