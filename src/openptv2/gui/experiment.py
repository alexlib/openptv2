"""
Experiment management for PyPTV

This module contains the Experiment class which manages parameter sets
and experiment configuration for PyPTV.
"""

import shutil
from pathlib import Path
from traits.api import HasTraits, Instance, List, Str, Bool, Any
from .parameter_manager import ParameterManager


class Paramset(HasTraits):
    """A parameter set identified by name and YAML file path.

    This is intentionally thin — no copy of parameters lives here.
    The authoritative parameters for any run live in its YAML file;
    the active run's parameters live in Experiment.pm.
    """
    name = Str()
    yaml_path = Path()

    def __init__(self, name: str, yaml_path: Path, **traits):
        super().__init__(**traits)
        self.name = name
        self.yaml_path = yaml_path


class Experiment(HasTraits):
    """
    The Experiment class manages parameter sets and experiment configuration.

    This is the main model class that owns all experiment data and parameters.
    It delegates parameter management to ParameterManager while handling
    the organization of multiple parameter sets.
    """
    active_params = Instance(Paramset)
    paramsets = List(Instance(Paramset))
    pm = Instance(ParameterManager)
    
    def __init__(self, pm: ParameterManager = None, **traits):
        super().__init__(**traits)
        self.paramsets = []
        self.pm = pm if pm is not None else ParameterManager()
        self._override_save_path = None
        # If pm has a loaded YAML path, add it as a paramset and set active
        yaml_path = getattr(self.pm, 'yaml_path', None)
        if yaml_path is not None:
            paramset = Paramset(name=yaml_path.stem, yaml_path=yaml_path)
            self.paramsets.append(paramset)
            self.active_params = paramset
        else:
            self.active_params = None

    def get_parameter(self, key):
        """Get parameter with ParameterManager delegation"""
        return self.pm.get_parameter(key)
    
    def load_parameters_for_active(self):
        """Load parameters from the active paramset's YAML into experiment.pm."""
        try:
            print(f"Loading parameters from YAML: {self.active_params.yaml_path}")
            self.pm.from_yaml(self.active_params.yaml_path)
        except Exception as e:
            raise IOError(f"Failed to load parameters from {self.active_params.yaml_path}: {e}")

    def save_active(self):
        """Save experiment.pm to the active paramset's YAML file (or override path)."""
        path = self._override_save_path or (self.active_params.yaml_path if self.active_params else None)
        if path is None:
            return
        self.pm.to_yaml(path)
        print(f"Parameters saved to {path}")

    def getParamsetIdx(self, paramset):
        """Get the index of a parameter set"""
        if isinstance(paramset, int):
            return paramset
        else:
            return self.paramsets.index(paramset)

    def addParamset(self, name: str, yaml_path: Path):
        """Add a new parameter set to the experiment"""
        # Ensure the YAML file exists, creating it from legacy directory if needed
        # if not yaml_path.exists():
        #     # Try to find legacy directory
        #     legacy_dir = yaml_path.parent / f"parameters{name}"
        #     if legacy_dir.exists() and legacy_dir.is_dir():
        #         print(f"Creating YAML from legacy directory: {legacy_dir}")
        #         pm = ParameterManager()
        #         pm.from_directory(legacy_dir)
        #         pm.to_yaml(yaml_path)
        #     else:
        #         print(f"Warning: Neither YAML file {yaml_path} nor legacy directory {legacy_dir} exists")

        # Create a simplified Paramset with just name and YAML path
        paramset = Paramset(name=name, yaml_path=yaml_path)
        self.paramsets.append(paramset)
        return paramset

    def removeParamset(self, paramset):
        """Remove a parameter set from the experiment"""
        paramset_idx = self.getParamsetIdx(paramset)
        
        paramset_obj = self.paramsets[paramset_idx]
        # Rename the YAML file to .bck
        yaml_path = getattr(paramset_obj, "yaml_path", None)
        if yaml_path and isinstance(yaml_path, Path) and yaml_path.exists():
            bck_path = yaml_path.with_suffix('.bck')
            yaml_path.rename(bck_path)
            print(f"Renamed YAML file to backup: {bck_path}")

        # Remove the corresponding legacy directory if it exists
        paramset_name = getattr(paramset_obj, 'name', '')
        if paramset_name and yaml_path:
            legacy_dir = yaml_path.parent / f"parameters{paramset_name}"
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
                print(f"Removed legacy directory: {legacy_dir}")

        self.paramsets.remove(self.paramsets[paramset_idx])

    def rename_paramset(self, old_name: str, new_name: str):
        """Rename a parameter set and its YAML file."""
        # Find the paramset by old_name
        paramset_obj = next((ps for ps in self.paramsets if ps.name == old_name), None)
        if paramset_obj is None:
            raise ValueError(f"No parameter set found with name '{old_name}'")

        old_yaml = paramset_obj.yaml_path
        if not old_yaml.exists():
            raise FileNotFoundError(f"YAML file for parameter set '{old_name}' does not exist: {old_yaml}")

        clean_new = new_name[11:] if new_name.startswith("parameters_") else new_name
        if old_yaml.name.startswith("parameters_"):
            new_yaml = old_yaml.parent / f"parameters_{clean_new}.yaml"
        else:
            new_yaml = old_yaml.parent / f"{clean_new}{old_yaml.suffix if old_yaml.suffix else '.yaml'}"

        if new_yaml.exists() and new_yaml != old_yaml:
            raise FileExistsError(f"YAML file for new name already exists: {new_yaml}")

        if new_yaml != old_yaml:
            old_yaml.rename(new_yaml)
            print(f"Renamed YAML file from {old_yaml} to {new_yaml}")

        # Update paramset object
        paramset_obj.name = clean_new
        paramset_obj.yaml_path = new_yaml

        return paramset_obj, new_yaml

    def nParamsets(self):
        """Get the number of parameter sets"""
        return len(self.paramsets)

    def set_active(self, paramset):
        """Set the active parameter set"""
        paramset_idx = self.getParamsetIdx(paramset)
        self.active_params = self.paramsets[paramset_idx]
        self.paramsets.pop(paramset_idx)
        self.paramsets.insert(0, self.active_params)
        # Load parameters for the newly active set
        self.load_parameters_for_active()

    def _collect_yaml_files(self, exp_path: Path):
        yaml_files = list(exp_path.glob("*parameters_*.yaml"))

        subdirs = [
            d for d in exp_path.iterdir()
            if d.is_dir() and d.name.startswith("parameters")
        ]

        for subdir in subdirs:
            run_name = subdir.name.replace("parameters", "") or "Run1"
            yaml_file = exp_path / f"parameters_{run_name}.yaml"

            if not yaml_file.exists():
                print(f"Converting legacy directory {subdir} to {yaml_file}")
                pm = ParameterManager()
                pm.from_directory(subdir)
                pm.to_yaml(yaml_file)

            yaml_files.append(yaml_file)

        result = sorted(set(yaml_files))
        if not result:
            raise FileNotFoundError(
                f"No parameter YAML files found in {exp_path} and no legacy "
                "parameter directories to convert. Create a parameters_<name>.yaml first."
            )
        return result

    def _run_name_from_yaml(self, yaml_file: Path):
        filename = yaml_file.stem
        if "parameters_" in filename:
            return filename.split("parameters_", 1)[1]
        if filename.startswith("parameters"):
            return filename[10:] or "Run1"
        if "_parameters" in filename:
            return filename.split("_parameters", 1)[0]
        return filename

    def _load_paramset_from_yaml(self, yaml_file: Path):
        run_name = self._run_name_from_yaml(yaml_file)
        print(f"Adding parameter set: {run_name} from {yaml_file}")
        return self.addParamset(run_name, yaml_file)

    # def export_legacy_directory(self, output_dir: Path):
    #     """Export current parameters to legacy .par files directory (for compatibility)"""
    #     if self.active_params is not None:
    #         self.pm.to_directory(output_dir)
    #         print(f"Exported parameters to legacy directory: {output_dir}")
    #     else:
    #         print("No active parameter set to export")

    def populate_runs(self, exp_path: Path, active_yaml: Path | None = None):
        """Populate parameter sets from an experiment directory.

        active_yaml: path to the YAML that should become the active paramset.
          When omitted the first discovered YAML is activated (alphabetical order).
        """
        self.paramsets = []

        yaml_files = self._collect_yaml_files(exp_path)

        for yaml_file in yaml_files:
            self._load_paramset_from_yaml(yaml_file)

        if self.nParamsets() == 0:
            return

        if active_yaml is not None:
            active_yaml = Path(active_yaml).resolve()
            for i, ps in enumerate(self.paramsets):
                if ps.yaml_path.resolve() == active_yaml:
                    self.set_active(i)
                    return
            print(
                f"WARNING: requested active YAML {active_yaml} not found among "
                "discovered parameter sets; falling back to first."
            )

        if self.active_params is None:
            self.set_active(0)


    def duplicate_paramset(self, run_name: str):
        """Duplicate a parameter set by copying its YAML file to a new file with '_copy' appended to the name."""
        # Find the paramset by name
        paramset_obj = next((ps for ps in self.paramsets if ps.name == run_name), None)
        if paramset_obj is None:
            raise ValueError(f"No parameter set found with name '{run_name}'")
        
        src_yaml = paramset_obj.yaml_path
        if not src_yaml.exists():
            raise FileNotFoundError(f"YAML file for parameter set '{run_name}' does not exist: {src_yaml}")
        
        # Create new name and path
        new_name = f"{run_name}_copy"
        new_yaml = src_yaml.parent / f"parameters_{new_name}.yaml"
        
        if new_yaml.exists():
            raise FileExistsError(f"Duplicate YAML file already exists: {new_yaml}")
        
        shutil.copy(src_yaml, new_yaml)
        print(f"Duplicated parameter set '{run_name}' to '{new_name}'")
        
        self.addParamset(new_name, new_yaml)
        return new_yaml

    def create_new_paramset(self, name: str, exp_path: Path, copy_from_active: bool = True):
        """Create a new parameter set YAML file"""
        yaml_file = exp_path / f"parameters_{name}.yaml"
        
        if yaml_file.exists():
            raise ValueError(f"Parameter set {name} already exists at {yaml_file}")
        
        if copy_from_active and self.active_params is not None:
            # Copy from active parameter set
            shutil.copy(self.active_params.yaml_path, yaml_file)
            print(f"Created new parameter set {name} by copying from {self.active_params.name}")
        
        self.addParamset(name, yaml_file)
        return yaml_file

    def delete_paramset(self, paramset):
        """Delete a parameter set, its YAML file, and corresponding legacy directory"""
        paramset_idx = self.getParamsetIdx(paramset)
        paramset_obj = self.paramsets[paramset_idx]

        # Ensure paramset_obj is a Paramset instance
        if not isinstance(paramset_obj, Paramset):
            raise TypeError("paramset_obj is not a Paramset instance")

        if paramset_obj == self.active_params:
            raise ValueError("Cannot delete the active parameter set")

        # Delete the YAML file
        yaml_path = getattr(paramset_obj, "yaml_path", None)
        if yaml_path and isinstance(yaml_path, Path) and yaml_path.exists():
            yaml_path.unlink()
            print(f"Deleted YAML file: {yaml_path}")

        # Delete corresponding legacy directory if it exists
        paramset_name = getattr(paramset_obj, 'name', '')
        if paramset_name and yaml_path:
            legacy_dir = yaml_path.parent / f"parameters{paramset_name}"
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
                print(f"Deleted legacy directory: {legacy_dir}")

        # Remove from list
        self.paramsets.remove(paramset_obj)
        print(f"Removed parameter set: {paramset_name}")

    def get_n_cam(self):
        """Get the global number of cameras"""
        return self.pm.get_n_cam()
