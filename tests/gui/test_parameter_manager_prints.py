from pathlib import Path

import yaml

from openptv2.gui.parameter_manager import ParameterManager


def test_print_cavity_yaml():
    pm = ParameterManager()
    pm.from_directory(
        str(
            Path(__file__).parent.parent.parent
            / "test_data"
            / "test_cavity"
            / "parameters"
        )
    )
    print("\n--- YAML for test_cavity ---")
    print(yaml.dump(pm.parameters, sort_keys=False, default_flow_style=False))


def test_print_splitter_yaml():
    pm = ParameterManager()
    pm.from_directory(
        str(
            Path(__file__).parent.parent.parent
            / "test_data"
            / "test_splitter"
            / "parameters"
        )
    )
    print("\n--- YAML for test_splitter ---")
    print(yaml.dump(pm.parameters, sort_keys=False, default_flow_style=False))
