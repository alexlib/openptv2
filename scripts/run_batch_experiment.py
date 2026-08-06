#!/usr/bin/env python3
r"""
OpenPTV2 Multi-Folder Batch Runner with Real-Time Progress, Anomaly Detection & Circuit Breaker
------------------------------------------------------------------------------------------------
Automates 2-step batch execution (Sample -> Quality Check -> Full Batch) across multiple run folders
with real-time progress updates, particle consistency monitoring, moving-average anomaly alerts,
live progress.json reporting, and manual abort controls.

Usage:
    uv run --project C:\Users\alex\projects\openptv2 python run_batch_experiment.py [--config batch_config.yaml] [--sample-only] [--skip-sample]
"""

import argparse
import datetime
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path
import yaml

class Term:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_print(msg: str, log_files=None, color=None):
    """Print message to terminal with optional color and append to log files."""
    display_msg = f"{color}{msg}{Term.ENDC}" if color else msg
    print(display_msg)
    
    clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
    if log_files:
        for lf in log_files:
            with open(lf, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {clean_msg}\n")

def verify_folder_setup(exp_root: Path, run_cfg: dict, shared_cal_dir: str):
    folder_name = run_cfg['folder']
    folder_path = exp_root / folder_name
    report = []
    is_valid = True
    info = {}
    
    if not folder_path.exists():
        report.append(f"Folder '{folder_name}' does not exist on disk!")
        return False, report, info

    img_dir = folder_path / "img"
    if not img_dir.exists():
        report.append(f"Image directory '{img_dir}' does not exist!")
        return False, report, info
        
    tifs = sorted(list(img_dir.glob("*.tif")))
    if not tifs:
        report.append(f"No .tif images found in '{img_dir}'!")
        return False, report, info
        
    info['tif_count'] = len(tifs)
    info['first_tif'] = tifs[0].name
    info['last_tif'] = tifs[-1].name
    
    report.append(f"Found {len(tifs)} raw .tif images ({tifs[0].name} ... {tifs[-1].name})")

    sample_yaml_path = folder_path / run_cfg['sample_yaml']
    batch_yaml_path = folder_path / run_cfg['batch_yaml']

    for label, y_path in [("Sample YAML", sample_yaml_path), ("Batch YAML", batch_yaml_path)]:
        if not y_path.exists():
            report.append(f"{label} file '{y_path.name}' is MISSING!")
            is_valid = False
            continue
            
        with open(y_path, 'r', encoding='utf-8') as f:
            y_data = yaml.safe_load(f)

        fixp = folder_path / y_data['cal_ori']['fixp_name']
        if not fixp.exists():
            report.append(f"Calibration fixp '{fixp}' referenced in {y_path.name} DOES NOT EXIST!")
            is_valid = False

        for ori_rel in y_data['cal_ori']['img_ori']:
            ori_p = folder_path / ori_rel
            if not ori_p.exists():
                report.append(f"Camera orientation file '{ori_p}' referenced in {y_path.name} DOES NOT EXIST!")
                is_valid = False

        base_tmpl = y_data['sequence']['base_name'][0]
        seq_first = y_data['sequence']['first']
        seq_last = y_data['sequence']['last']
        
        first_frame_path = folder_path / (base_tmpl % seq_first)
        last_frame_path = folder_path / (base_tmpl % seq_last)

        if not first_frame_path.exists():
            report.append(f"First frame {seq_first} ({first_frame_path.name}) specified in {y_path.name} NOT FOUND on disk!")
            is_valid = False
            
        if not last_frame_path.exists():
            report.append(f"Last frame {seq_last} ({last_frame_path.name}) specified in {y_path.name} NOT FOUND on disk!")
            is_valid = False
            
        report.append(f"Verified {y_path.name}: sequence template '{base_tmpl}', range {seq_first}..{seq_last}")

    return is_valid, report, info

def run_and_monitor_openptv2_batch(yaml_path: Path, log_file: Path, anomaly_cfg: dict, exp_root: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    folder_dir = yaml_path.parent
    progress_json_path = folder_dir / "progress.json"

    abort_trigger_name = anomaly_cfg.get("abort_file_trigger", "ABORT_RUN")
    folder_abort_file = folder_dir / abort_trigger_name
    root_abort_file = exp_root / abort_trigger_name

    for f_p in [folder_abort_file, root_abort_file]:
        if f_p.exists():
            try: f_p.unlink()
            except Exception: pass

    max_drop_pct = anomaly_cfg.get("max_particle_drop_pct", 30.0)
    max_spike_pct = anomaly_cfg.get("max_particle_spike_pct", 50.0)
    min_particles = anomaly_cfg.get("min_particles_threshold", 5)
    stop_on_anomaly = anomaly_cfg.get("stop_on_anomaly", False)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        y_data = yaml.safe_load(f)
    first_frame = y_data.get('sequence', {}).get('first', 1)
    last_frame = y_data.get('sequence', {}).get('last', 10)
    total_frames = max(1, last_frame - first_frame + 1)

    # Resolve executable
    openptv_proj = Path("C:/Users/alex/projects/openptv2")
    venv_exe = openptv_proj / ".venv" / "Scripts" / "openptv2-batch.exe"
    if venv_exe.exists():
        cmd = [str(venv_exe), str(yaml_path)]
    else:
        cmd = ["uv", "run", "--project", str(openptv_proj), "openptv2-batch", str(yaml_path)]
    
    corr_pattern = re.compile(r'Frame (\d+) had \[(\d+),\s*(\d+),\s*(\d+)\] correspondences')
    step_pattern = re.compile(r'(?:track3d\s+)?step:\s*(\d+),\s*curr:\s*(\d+),\s*next:\s*(\d+),\s*links:\s*(\d+)')
    
    recent_particles = []
    anomalies = []
    current_stage = "INITIALIZING"
    output_lines = []

    t_start = time.time()

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== OpenPTV2 Monitored Batch Started at {datetime.datetime.now()} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"Working Dir: {folder_dir}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(folder_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            output_lines.append(line)

            if folder_abort_file.exists() or root_abort_file.exists():
                log_print(f"\n{Term.FAIL}[ABORT TRIGGER] Detected abort signal file ({abort_trigger_name})! Stopping run...{Term.ENDC}", [log_file])
                proc.terminate()
                anomalies.append({"frame": -1, "message": "Manual abort trigger activated"})
                break

            if "Running sequence plugin" in line:
                current_stage = "DETECTION_AND_CORRESPONDENCE"
            elif "Running tracking plugin" in line or "Running Standard Forward Tracking" in line:
                current_stage = "3D_TRACKING"

            m_corr = corr_pattern.search(line)
            if m_corr:
                frame_num = int(m_corr.group(1))
                c4, c3, c2 = map(int, m_corr.groups()[1:])
                total_parts = c4 + c3 + c2

                moving_avg = sum(recent_particles[-10:]) / len(recent_particles[-10:]) if recent_particles else float(total_parts)
                recent_particles.append(total_parts)

                elapsed = time.time() - t_start
                frames_done = len(recent_particles)
                pct = min(100.0, (frames_done / total_frames) * 100.0)
                fps = frames_done / max(elapsed, 0.001)
                eta_sec = (total_frames - frames_done) / fps if fps > 0 else 0

                is_anomaly = False
                anomaly_msg = ""

                if total_parts < min_particles:
                    is_anomaly = True
                    anomaly_msg = f"Particle count ({total_parts}) below minimum threshold ({min_particles})!"
                elif len(recent_particles) > 5 and moving_avg > 0:
                    pct_change = ((total_parts - moving_avg) / moving_avg) * 100.0
                    if pct_change < -max_drop_pct:
                        is_anomaly = True
                        anomaly_msg = f"Particle count dropped by {abs(pct_change):.1f}% (curr={total_parts}, avg={moving_avg:.1f})!"
                    elif pct_change > max_spike_pct:
                        is_anomaly = True
                        anomaly_msg = f"Particle count spiked by {pct_change:.1f}% (curr={total_parts}, avg={moving_avg:.1f})!"

                if is_anomaly:
                    anomalies.append({"frame": frame_num, "message": anomaly_msg})
                    log_print(f"  {Term.WARNING}[ANOMALY ALERT @ Frame {frame_num}] {anomaly_msg}{Term.ENDC}", [log_file])
                    if stop_on_anomaly:
                        log_print(f"{Term.FAIL}[CIRCUIT BREAKER] Aborting run due to severe anomaly.{Term.ENDC}", [log_file])
                        proc.terminate()
                        break

                if frames_done % 10 == 0 or frames_done == 1 or frames_done == total_frames:
                    log_print(f"  [{current_stage}] Frame {frame_num} ({frames_done}/{total_frames} | {pct:.1f}%) | Particles: {total_parts} (avg: {moving_avg:.1f}) | {fps:.1f} fps | ETA: {eta_sec:.0f}s", [log_file], color=Term.OKCYAN)

                prog_data = {
                    "stage": current_stage,
                    "current_frame": frame_num,
                    "frames_completed": frames_done,
                    "total_frames": total_frames,
                    "progress_pct": round(pct, 1),
                    "particles": total_parts,
                    "moving_avg_particles": round(moving_avg, 1),
                    "anomalies_count": len(anomalies),
                    "anomalies": anomalies[-5:],
                    "status": "RUNNING",
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(eta_sec, 1),
                    "updated_at": datetime.datetime.now().isoformat()
                }
                with open(progress_json_path, "w", encoding="utf-8") as pf:
                    json.dump(prog_data, pf, indent=2)

            m_step = step_pattern.search(line)
            if m_step:
                step_num, curr_p, next_p, links = map(int, m_step.groups())
                current_stage = f"TRACKING_STEP_{step_num}"
                if step_num % 10 == 0 or step_num == 1:
                    log_print(f"  [3D_TRACKING Step {step_num}] Particles: {curr_p} -> Active Links: {links}", [log_file], color=Term.OKBLUE)

        proc.wait()

    final_status = "SUCCESS" if proc.returncode == 0 and not anomalies else ("ABORTED" if anomalies and stop_on_anomaly else "COMPLETED_WITH_WARNINGS")
    prog_data = {
        "stage": "COMPLETED",
        "progress_pct": 100.0,
        "anomalies_count": len(anomalies),
        "anomalies": anomalies,
        "status": final_status,
        "elapsed_sec": round(time.time() - t_start, 1),
        "updated_at": datetime.datetime.now().isoformat()
    }
    with open(progress_json_path, "w", encoding="utf-8") as pf:
        json.dump(prog_data, pf, indent=2)

    return proc.returncode, output_lines

def parse_batch_output(output_lines):
    metrics = {
        'correspondences': [],
        'avg_particles': 0.0,
        'avg_links': 0.0,
        'total_links': 0,
        'completed': False
    }

    corr_pattern = re.compile(r'Frame \d+ had \[(\d+),\s*(\d+),\s*(\d+)\] correspondences')
    avg_pattern = re.compile(r'Average over sequence, particles:\s*([\d\.]+),\s*links:\s*([\d\.]+)')
    hybrid_pattern = re.compile(r'Hybrid 3D\+Corr Tracking.*avg links/step = ([\d\.]+)')
    step_pattern = re.compile(r'(?:track3d\s+)?step:\s*\d+,\s*curr:\s*(\d+),\s*next:\s*\d+,\s*links:\s*(\d+)')
    links_pattern = re.compile(r'Post-process links:\s*\d+\s*->\s*(\d+)')

    detected_particles = []
    step_links = []

    for line in output_lines:
        if 'Batch processing completed successfully' in line or 'Tracking' in line:
            metrics['completed'] = True

        m_corr = corr_pattern.search(line)
        if m_corr:
            c4, c3, c2 = map(int, m_corr.groups())
            metrics['correspondences'].append((c4, c3, c2))

        m_step = step_pattern.search(line)
        if m_step:
            curr_p, lks = map(int, m_step.groups())
            detected_particles.append(curr_p)
            step_links.append(lks)

        m_hybrid = hybrid_pattern.search(line)
        if m_hybrid:
            metrics['avg_links'] = float(m_hybrid.group(1))

        m_avg = avg_pattern.search(line)
        if m_avg:
            metrics['avg_particles'] = float(m_avg.group(1))
            metrics['avg_links'] = float(m_avg.group(2))

        m_links = links_pattern.search(line)
        if m_links:
            metrics['total_links'] = int(m_links.group(1))

    if detected_particles:
        metrics['avg_particles'] = float(sum(detected_particles) / len(detected_particles))
    if step_links:
        metrics['total_links'] = sum(step_links)
        metrics['avg_links'] = float(metrics['total_links'] / len(step_links))

    if metrics['total_links'] == 0 and metrics['avg_links'] > 0 and len(metrics['correspondences']) > 0:
        metrics['total_links'] = int(metrics['avg_links'] * (len(metrics['correspondences']) - 1))

    return metrics

def main():
    parser = argparse.ArgumentParser(description="OpenPTV2 Monitored Batch Processing Runner")
    parser.add_argument("--config", default="batch_config.yaml", help="Path to batch_config.yaml")
    parser.add_argument("--sample-only", action="store_true", help="Run step 1 (sample) only")
    parser.add_argument("--skip-sample", action="store_true", help="Skip step 1 and run full batch directly")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        if (script_dir / config_path).exists():
            config_path = (script_dir / config_path).resolve()
        else:
            config_path = config_path.resolve()

    if not config_path.exists():
        print(f"{Term.FAIL}Error: Config file '{config_path}' not found!{Term.ENDC}")
        sys.exit(1)

    exp_root = config_path.parent
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", exp_root.name)
    runs = cfg.get("runs", [])
    qgate = cfg.get("quality_gate", {})
    anomaly_cfg = cfg.get("anomaly_detection", {})
    exec_cfg = cfg.get("execution", {})

    log_dir = exp_root / exec_cfg.get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    master_log = log_dir / f"master_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_print(f"============================================================", color=Term.HEADER)
    log_print(f"  OpenPTV2 Monitored Batch Processing - Experiment: {exp_name}", log_files=[master_log], color=Term.BOLD)
    log_print(f"============================================================", color=Term.HEADER)
    log_print(f"Loaded config: {config_path.name}", log_files=[master_log])
    log_print(f"Target runs: {[r['name'] for r in runs]}\n", log_files=[master_log])

    # Step 0: Pre-flight Verification
    log_print(">>> Step 0: Verifying internal folder configurations & disk images...", log_files=[master_log], color=Term.OKCYAN)
    all_verified = True
    
    for run in runs:
        log_print(f"\n[Run: {run['name']}] Checking folder '{run['folder']}'...", log_files=[master_log])
        ok, report_lines, info = verify_folder_setup(exp_root, run, cfg.get("shared_calibration_dir", ""))
        for line in report_lines:
            status_color = Term.OKGREEN if ok else Term.FAIL
            log_print(f"  - {line}", log_files=[master_log], color=status_color)
        if not ok:
            all_verified = False

    if not all_verified:
        log_print(f"\n{Term.FAIL}Pre-flight verification failed! Please fix errors listed above before running.{Term.ENDC}", log_files=[master_log])
        sys.exit(1)

    log_print(f"\n{Term.OKGREEN}Pre-flight verification PASSED for all {len(runs)} folders!{Term.ENDC}\n", log_files=[master_log])

    # Step 1: Small / Sample Run
    sample_passed = True
    sample_results = {}

    if not args.skip_sample:
        log_print("============================================================", color=Term.HEADER)
        log_print("  Step 1: Running Small Subset / Sample Check", log_files=[master_log], color=Term.BOLD)
        log_print("============================================================", color=Term.HEADER)

        for run in runs:
            r_name = run['name']
            folder_p = exp_root / run['folder']
            sample_yaml = folder_p / run['sample_yaml']
            run_log = log_dir / f"{r_name}_sample.log"
            per_folder_log = folder_p / "pipeline_sample.log"

            log_print(f"\n[{r_name}] Launching sample run: {sample_yaml.name} ...", log_files=[master_log, run_log], color=Term.OKBLUE)
            
            t0 = time.time()
            ret_code, out_lines = run_and_monitor_openptv2_batch(sample_yaml, run_log, anomaly_cfg, exp_root)
            elapsed = time.time() - t0

            with open(run_log, 'r', encoding='utf-8') as rf, open(per_folder_log, 'w', encoding='utf-8') as pf:
                pf.write(rf.read())

            if ret_code != 0:
                log_print(f"[{r_name}] {Term.FAIL}Sample run FAILED with exit code {ret_code}! See log: {run_log}{Term.ENDC}", log_files=[master_log, run_log])
                sample_passed = False
                continue

            metrics = parse_batch_output(out_lines)
            sample_results[r_name] = metrics

            avg_m = metrics['avg_particles']
            tot_l = metrics['total_links']
            n_frames = len(metrics['correspondences'])

            log_print(f"[{r_name}] Sample run completed in {elapsed:.2f}s ({n_frames} frames)", log_files=[master_log, run_log])
            log_print(f"  - Avg particles/frame: {avg_m:.1f}", log_files=[master_log, run_log])
            log_print(f"  - Avg links/frame: {metrics['avg_links']:.1f}", log_files=[master_log, run_log])
            log_print(f"  - Total links: {tot_l}", log_files=[master_log, run_log])

            min_matches = qgate.get("min_matches_per_frame", 10)
            min_tracks = qgate.get("min_tracks_created", 10)

            if avg_m < min_matches or tot_l < min_tracks or not metrics['completed']:
                log_print(f"  - {Term.FAIL}QUALITY GATE FAILED for {r_name}:{Term.ENDC} (matches={avg_m:.1f} < {min_matches} or links={tot_l} < {min_tracks})", log_files=[master_log, run_log])
                sample_passed = False
            else:
                log_print(f"  - {Term.OKGREEN}QUALITY GATE PASSED for {r_name}!{Term.ENDC}", log_files=[master_log, run_log])

        if args.sample_only:
            log_print(f"\n{Term.OKCYAN}Sample run completed (--sample-only specified). Stopping here.{Term.ENDC}", log_files=[master_log])
            return

        if not sample_passed:
            log_print(f"\n{Term.FAIL}!!! Quality Gate FAILED for one or more sample runs. Aborting full batch processing. !!!{Term.ENDC}", log_files=[master_log])
            sys.exit(1)

        log_print(f"\n{Term.OKGREEN}>>> All sample runs passed quality gates! Proceeding to Step 2 (Full Batch Run)...{Term.ENDC}\n", log_files=[master_log])

    # Step 2: Full / Batch Run
    log_print("============================================================", color=Term.HEADER)
    log_print("  Step 2: Executing Full Monitored Batch Run", log_files=[master_log], color=Term.BOLD)
    log_print("============================================================", color=Term.HEADER)

    batch_summary = {}

    for run in runs:
        r_name = run['name']
        folder_p = exp_root / run['folder']
        batch_yaml = folder_p / run['batch_yaml']
        run_log = log_dir / f"{r_name}_batch.log"
        per_folder_log = folder_p / "pipeline_batch.log"

        log_print(f"\n[{r_name}] Launching full batch run: {batch_yaml.name} ...", log_files=[master_log, run_log], color=Term.OKBLUE)
        
        t0 = time.time()
        ret_code, out_lines = run_and_monitor_openptv2_batch(batch_yaml, run_log, anomaly_cfg, exp_root)
        elapsed = time.time() - t0

        with open(run_log, 'r', encoding='utf-8') as rf, open(per_folder_log, 'w', encoding='utf-8') as pf:
            pf.write(rf.read())

        if ret_code != 0:
            log_print(f"[{r_name}] {Term.FAIL}Full batch run FAILED with exit code {ret_code}! See log: {run_log}{Term.ENDC}", log_files=[master_log, run_log])
            batch_summary[r_name] = {'success': False, 'time': elapsed}
            continue

        metrics = parse_batch_output(out_lines)
        batch_summary[r_name] = {
            'success': True,
            'time': elapsed,
            'avg_particles': metrics['avg_particles'],
            'total_links': metrics['total_links'],
            'frames': len(metrics['correspondences'])
        }

        log_print(f"[{r_name}] {Term.OKGREEN}Full batch completed successfully in {elapsed:.2f}s ({elapsed/60.0:.2f} min){Term.ENDC}", log_files=[master_log, run_log])
        log_print(f"  - Frames processed: {len(metrics['correspondences'])}", log_files=[master_log, run_log])
        log_print(f"  - Avg particles/frame: {metrics['avg_particles']:.1f}", log_files=[master_log, run_log])
        log_print(f"  - Total links: {metrics['total_links']}", log_files=[master_log, run_log])

    # Final Summary Table
    log_print("\n============================================================", color=Term.HEADER)
    log_print("  Final Multi-Folder Batch Execution Summary", log_files=[master_log], color=Term.BOLD)
    log_print("============================================================", color=Term.HEADER)
    log_print(f"{'Run Folder':<12} | {'Status':<10} | {'Frames':<8} | {'Avg Particles':<14} | {'Total Links':<12} | {'Time (min)':<10}", log_files=[master_log])
    log_print("-" * 78, log_files=[master_log])

    all_success = True
    for r_name, res in batch_summary.items():
        if res['success']:
            log_print(f"{r_name:<12} | SUCCESS    | {res['frames']:<8} | {res['avg_particles']:<14.1f} | {res['total_links']:<12} | {res['time']/60.0:<10.2f}", log_files=[master_log])
        else:
            all_success = False
            log_print(f"{r_name:<12} | FAILED     | {'N/A':<8} | {'N/A':<14} | {'N/A':<12} | {res['time']/60.0:<10.2f}", log_files=[master_log])

    log_print("-" * 78, log_files=[master_log])
    if all_success:
        log_print(f"\n{Term.OKGREEN}ALL RUNS COMPLETED SUCCESSFULLY! Output written to each folder's res/ directory.{Term.ENDC}\n", log_files=[master_log])
    else:
        log_print(f"\n{Term.FAIL}One or more runs failed. Check log files in {log_dir} for details.{Term.ENDC}\n", log_files=[master_log])

if __name__ == "__main__":
    main()
