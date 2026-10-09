"""Stage the 10-frame audit run: harmonized params + img/ symlink farm.

For each rig, writes parameters_audit10.yaml (copy of parameters_Run1.yaml
with ONLY these changes, all else byte-identical):
  * bubble_detection.z_thresh -> 6.0        (was 6.5 on rig 1-4)
  * criteria cn/cnx/cny -> 0.01             (were 0.02 on rig 5-8)
  * track.dacc -> 5.0                       (was 5.5 on rig 5-8)
  * targ_rec block -> rig 1-4 values        (legacy; bubble_sequence ignores it)
  * sequence.base_name -> img/camN/%08d.tiff, first 10200, last 10209
and symlinks img/cam{1..4}/0001020{F}.tiff -> ../../Messung_N/<frame>_HASH.tiff
for frames 10200..10209 (the 8-digit filename prefix IS the frame number).

Production parameters_Run1.yaml files are never touched.

Usage:
    uv run python scripts/illmenau/stage_audit_run.py
"""

from pathlib import Path

import yaml

RAW = Path("/Users/alex/Downloads/Ilmenau")
RIGS = {
    "openptv_illmenau_4cam": [1, 2, 3, 4],
    "openptv_illmenau_5678": [5, 6, 7, 8],
}
FIRST, LAST = 10200, 10209


def main():
    # canonical targ_rec from rig 1-4 (legacy block, kept identical)
    ref = yaml.safe_load(
        (RAW / "openptv_illmenau_4cam" / "parameters_Run1.yaml").read_text())
    for rig, cams in RIGS.items():
        d = RAW / rig
        y = yaml.safe_load((d / "parameters_Run1.yaml").read_text())
        y["bubble_detection"]["z_thresh"] = 6.0
        for k in ("cn", "cnx", "cny"):
            y["criteria"][k] = 0.01
        y["track"]["dacc"] = 5.0
        y["targ_rec"] = ref["targ_rec"]
        y["sequence"]["base_name"] = [
            f"img/cam{i + 1}/%08d.tiff" for i in range(4)]
        y["sequence"]["first"] = FIRST
        y["sequence"]["last"] = LAST
        with open(d / "parameters_audit10.yaml", "w") as fh:
            yaml.safe_dump(y, fh, sort_keys=False)
        print(f"{rig}: wrote parameters_audit10.yaml", flush=True)
        for i, n in enumerate(cams):
            files = sorted((RAW / f"Messung_{n}").glob("*.tiff"))
            byframe = {int(p.name.split("_")[0]): p for p in files}
            dest = d / "img" / f"cam{i + 1}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in range(FIRST, LAST + 1):
                src = byframe[f]
                link = dest / f"{f:08d}.tiff"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(f"../../../Messung_{n}/{src.name}")
        print(f"{rig}: img/ symlinks for {FIRST}..{LAST}", flush=True)


if __name__ == "__main__":
    main()
