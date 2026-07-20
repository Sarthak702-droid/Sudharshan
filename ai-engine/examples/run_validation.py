import json
import os
import math
import numpy as np

def calculate_angular_error(deg1, deg2):
    diff = abs(deg1 - deg2)
    return min(diff, 360.0 - diff)

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def run_validation(metrics_path, labels_path, report_path):
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} does not exist. Run pipeline first.")
        return
    if not os.path.exists(labels_path):
        print(f"Labels file {labels_path} does not exist. Run generate_mock_labels.py first.")
        return

    with open(metrics_path, "r") as f:
        metrics_records = json.load(f)
    with open(labels_path, "r") as f:
        labels_data = json.load(f)

    # Group metrics by frame_id and grid_id
    preds = {}
    for r in metrics_records:
        fid = str(r["frame_id"])
        gid = r["grid_id"]
        preds[(fid, gid)] = r

    # Process matches
    count_errors = []
    angular_errors = []
    speed_errors = []

    # Event confusion matrices: [TP, FP, FN, TN]
    events = {
        "counter_flow": [0, 0, 0, 0],
        "stagnation": [0, 0, 0, 0],
        "speed_surge": [0, 0, 0, 0],
        "turbulence": [0, 0, 0, 0]
    }

    labels_frames = labels_data.get("frames", {})
    matched_count = 0

    for fid, grid_anns in labels_frames.items():
        for ann in grid_anns:
            gid = ann["grid_id"]
            if (fid, gid) in preds:
                pred = preds[(fid, gid)]
                matched_count += 1

                # Density/Count Error
                gt_count = ann["count_estimate"]
                pred_count = pred["crowd_count"]
                count_errors.append(pred_count - gt_count)

                # Speed Error
                # Check speed level mapping
                gt_speed_level = ann["motion_level"]
                pred_speed = pred["motion_magnitude"]
                pred_speed_level = pred["speed_level"]

                # Direction Error
                if gt_count > 0.1 and pred_count > 0.1:
                    gt_deg = ann["dominant_direction_deg"]
                    pred_deg = pred["direction_deg"]
                    angular_errors.append(calculate_angular_error(gt_deg, pred_deg))

                # Event Logic check
                # 1. Counter Flow
                gt_cf = ann["counter_flow"]
                pred_cf = pred["flow_conflict"]
                if gt_cf and pred_cf: events["counter_flow"][0] += 1  # TP
                elif not gt_cf and pred_cf: events["counter_flow"][1] += 1  # FP
                elif gt_cf and not pred_cf: events["counter_flow"][2] += 1  # FN
                else: events["counter_flow"][3] += 1  # TN

                # 2. Stagnation
                gt_st = ann["stagnation"]
                pred_st = pred["stasis_warning"]
                if gt_st and pred_st: events["stagnation"][0] += 1  # TP
                elif not gt_st and pred_st: events["stagnation"][1] += 1  # FP
                elif gt_st and not pred_st: events["stagnation"][2] += 1  # FN
                else: events["stagnation"][3] += 1  # TN

                # 3. Speed Surge
                gt_ss = ann["collective_speed_surge"]
                pred_ss = pred["speed_surge_warning"]
                if gt_ss and pred_ss: events["speed_surge"][0] += 1  # TP
                elif not gt_ss and pred_ss: events["speed_surge"][1] += 1  # FP
                elif gt_ss and not pred_ss: events["speed_surge"][2] += 1  # FN
                else: events["speed_surge"][3] += 1  # TN

                # 4. Turbulence
                gt_tb = ann["turbulence"]
                pred_tb = pred["turbulence_warning"]
                if gt_tb and pred_tb: events["turbulence"][0] += 1  # TP
                elif not gt_tb and pred_tb: events["turbulence"][1] += 1  # FP
                elif gt_tb and not pred_tb: events["turbulence"][2] += 1  # FN
                else: events["turbulence"][3] += 1  # TN

    if matched_count == 0:
        print("No matches found between labels and metrics. Check frame IDs or sequence names.")
        return

    # Compute aggregate stats
    count_errors = np.array(count_errors)
    mae = float(np.mean(np.abs(count_errors)))
    rmse = float(math.sqrt(np.mean(count_errors ** 2)))

    mean_ang_err = float(np.mean(angular_errors)) if angular_errors else 0.0
    med_ang_err = float(np.median(angular_errors)) if angular_errors else 0.0

    ang_acc_15 = float(np.mean([e <= 15.0 for e in angular_errors])) if angular_errors else 0.0
    ang_acc_30 = float(np.mean([e <= 30.0 for e in angular_errors])) if angular_errors else 0.0

    print("==========================================================")
    print("                SUDHARSHAN VALIDATION REPORT              ")
    print("==========================================================")
    print(f"Matched Frame-Grids: {matched_count}")
    print(f"Count Metrics:")
    print(f"  MAE:  {mae:.4f} people")
    print(f"  RMSE: {rmse:.4f} people")
    print(f"Direction Metrics:")
    print(f"  Mean Angular Error:   {mean_ang_err:.2f}°")
    print(f"  Median Angular Error:  {med_ang_err:.2f}°")
    print(f"  Accuracy (<=15°):      {ang_acc_15*100.1:.2f}%")
    print(f"  Accuracy (<=30°):      {ang_acc_30*100.1:.2f}%")

    report_content = f"""# Sudharshan Core Detection Validation Report

This report summarizes the scientific accuracy and mathematical validity of **Sudharshan** (Crowd Flow Engine) evaluated against ground-truth labels.

## 1. Evaluation Summary
* **Sequence ID**: {labels_data.get('sequence_id', 'Unknown')}
* **Total Evaluated Grid Cells**: {matched_count}

## 2. Density & Count Metrics
* **Mean Absolute Error (MAE)**: {mae:.4f} persons/grid
* **Root Mean Squared Error (RMSE)**: {rmse:.4f} persons/grid

## 3. Directional Motion Metrics
* **Mean Angular Error**: {mean_ang_err:.2f}°
* **Median Angular Error**: {med_ang_err:.2f}°
* **Angular Accuracy ($\le 15°$)**: {ang_acc_15*100.0:.2f}%
* **Angular Accuracy ($\le 30°$)**: {ang_acc_30*100.0:.2f}%

## 4. Event Logic Detections (Confusion Matrices)

| Event Type | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for name, matrix in events.items():
        tp, fp, fn, tn = matrix
        p, r, f1 = calculate_f1(tp, fp, fn)
        print(f"Event: {name.upper()}:")
        print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        report_content += f"| **{name.upper()}** | {tp} | {fp} | {fn} | {p:.2%} | {r:.2%} | {f1:.2%} |\n"

    report_content += """
---
*Report generated automatically on validation run.*
"""

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Saved validation report to: {report_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_file = os.path.join(base_dir, "Sudharshan", "outputs", "grid_metrics.json")
    labels_file = os.path.join(base_dir, "Sudharshan", "configs", "labels", "visdrone_uav0000339_labels.json")
    report_file = os.path.join(base_dir, "docs", "CORE_VALIDATION_REPORT.md")

    run_validation(metrics_file, labels_file, report_file)
