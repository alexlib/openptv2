import copy
import yaml
from pathlib import Path
import subprocess

yaml_path = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\parameters_wp1_sample.yaml")
y_orig = yaml.safe_load(yaml_path.read_text())

openptv_proj = Path("C:/Users/alex/projects/openptv2")

results = []

for v_max in [1.9, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    for acc in [1.9, 3.0, 5.0, 8.0]:
        y_test = copy.deepcopy(y_orig)
        y_test['track']['dvxmin'] = -v_max
        y_test['track']['dvxmax'] = v_max
        y_test['track']['dvymin'] = -v_max
        y_test['track']['dvymax'] = v_max
        y_test['track']['dvzmin'] = -v_max
        y_test['track']['dvzmax'] = v_max
        y_test['track']['dacc'] = acc

        yaml_path.write_text(yaml.safe_dump(y_test, sort_keys=False))

        cmd = ["uv", "run", "--project", str(openptv_proj), "openptv2-batch", str(yaml_path)]
        res = subprocess.run(cmd, cwd=yaml_path.parent, capture_output=True, text=True)

        # Parse average over sequence line
        for line in res.stdout.splitlines():
            if "Average over sequence" in line:
                print(f"v_max={v_max:4.1f}, acc={acc:4.1f} -> {line.strip()}")
                results.append((v_max, acc, line.strip()))

# Restore original yaml
yaml_path.write_text(yaml.safe_dump(y_orig, sort_keys=False))
