import json
import os
import argparse
import re
from src.events.parser import parse_wtts

def compute_item_precisions(predictions, gt_path):
    ground_truth = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = str(data.get('document_id'))
            if doc_id:
                gt_dict = {ann['item']: str(ann['ground_truth']).strip().lower() for ann in data.get('annotations', [])}
                ground_truth[doc_id] = gt_dict

    item_stats = {}

    for doc_id, pred_dict in predictions.items():
        if doc_id not in ground_truth:
            continue
        gt_dict = ground_truth[doc_id]

        for item, gt_val in gt_dict.items():
            pred_data = pred_dict.get(item, 'unknown')
            if isinstance(pred_data, dict):
                pred_val = str(pred_data.get('value', 'unknown')).strip().lower()
            else:
                pred_val = str(pred_data).strip().lower()

            if item not in item_stats:
                item_stats[item] = {'correct': 0, 'predicted': 0}

            # Only track precision for y/n predictions
            if pred_val in ['y', 'n']:
                item_stats[item]['predicted'] += 1
                if pred_val == gt_val:
                    item_stats[item]['correct'] += 1

    item_precisions = {}
    for item, stats in item_stats.items():
        if stats['predicted'] > 0:
            item_precisions[item] = stats['correct'] / stats['predicted']
        else:
            item_precisions[item] = 0.5 # Default neutral precision if never predicted

    return item_precisions

def calibrate_predictions(preds_path="data/processed/predictions.json", gt_path="data/raw/dev_gt.jsonl", threshold=0.45):
    
    print("[CALIBRATE] Loading predictions...")
    with open(preds_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print("[CALIBRATE] Computing item-level historical precision...")
    item_precisions = compute_item_precisions(predictions, gt_path)

    calibrated = {}
    total_reverted = 0
    total_evaluated = 0

    print("[CALIBRATE] Applying fusion scoring logic...")
    for pid, pred_dict in predictions.items():
        
        # Load WTTS to get Evidence Strengths (W)
        wtts_path = f"data/processed/materialized_ehr/{pid}.jsonl"
        sid_weights = {}
        
        if os.path.exists(wtts_path):
            with open(wtts_path, 'r', encoding='utf-8') as f:
                wtts_data = json.load(f)
                wtts_string = wtts_data.get("text", "")
                events = parse_wtts(wtts_string)
                for ev, ts, P_j, W, sid in events:
                    if sid:
                        sid_weights[sid] = W

        calibrated[pid] = {}

        for item, pred_data in pred_dict.items():
            
            if not isinstance(pred_data, dict):
                # Old format, leave as is
                calibrated[pid][item] = pred_data
                continue
                
            val = pred_data.get("value", "unknown")
            evidence = pred_data.get("evidence", [])
            
            if val not in ['y', 'n']:
                calibrated[pid][item] = pred_data
                continue

            total_evaluated += 1

            # 1. Item Precision
            P_item = item_precisions.get(item, 0.5)

            # 2. Model Confidence
            # 1 piece of evidence = 0.5, 2+ pieces = 1.0
            C_model = min(1.0, len(evidence) * 0.5)

            # 3. Evidence Strength
            E_str = 0.2 # Default low score if evidence tag isn't found
            weights = []
            for ev_tag in evidence:
                # ev_tag might be "[S_12]" or "S_12"
                clean_tag = ev_tag.replace("[", "").replace("]", "").strip()
                if clean_tag in sid_weights:
                    weights.append(sid_weights[clean_tag])
                # try adding brackets back just in case parser kept them
                elif f"[{clean_tag}]" in sid_weights:
                    weights.append(sid_weights[f"[{clean_tag}]"])

            if weights:
                E_str = max(weights) # Use the strongest piece of evidence provided

            # Gating Logic (Waterfall Filter)
            is_unknown = False
            
            if P_item < 0.15:
                is_unknown = True
            elif C_model < 0.5:
                is_unknown = True
            elif E_str < 0.2:
                is_unknown = True

            if is_unknown:
                # Revert to unknown
                calibrated[pid][item] = {
                    "value": "unknown",
                    "evidence": [],
                    "_calibration_metrics": f"P={P_item:.2f}, C={C_model:.2f}, E={E_str:.2f}"
                }
                total_reverted += 1
            else:
                # Keep prediction
                pred_data["_calibration_metrics"] = f"P={P_item:.2f}, C={C_model:.2f}, E={E_str:.2f}"
                calibrated[pid][item] = pred_data

    out_path = "data/processed/calibrated_predictions.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(calibrated, f, indent=4)

    print(f"\n[SUCCESS] Calibration Complete!")
    print(f"-> Total Weak Predictions Reverted to Unknown: {total_reverted}/{total_evaluated}")
    print(f"-> Saved to: {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="data/processed/predictions.json")
    parser.add_argument("--gt", default="data/raw/dev_gt.jsonl")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    calibrate_predictions(args.preds, args.gt, args.threshold)
