from openptv2.gui import cli


def test_cli_template():
    assert cli.cli() == "CLI template"
