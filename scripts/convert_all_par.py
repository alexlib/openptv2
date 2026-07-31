import sys
from pathlib import Path

# Add the project root to sys.path to resolve openptv2/gui imports
sys.path.insert(0, str(Path(".").resolve()))

from gui.pyptv.parameter_manager import ParameterManager


def convert_dir(dir_path):
    print(f"Converting {dir_path}...")
    pm = ParameterManager()
    pm.from_directory(dir_path)

    # Generate the yaml file next to it
    if dir_path.name == "parameters":
        run_name = "Run1"
    else:
        run_name = dir_path.name.replace("parameters", "")
        if not run_name:
            run_name = "Run1"

    yaml_file = dir_path.parent / f"parameters_{run_name}.yaml"
    pm.to_yaml(yaml_file)
    print(f"  -> {yaml_file}")

if __name__ == "__main__":
    test_data = Path("test_data")
    for p in test_data.rglob("parameters*"):
        if p.is_dir():
            convert_dir(p)
