import json
import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_predictions(predictions_path="data/processed/predictions.json", gt_path="data/raw/dev_gt.jsonl"):
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at {predictions_path}")
        return
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}")
        return

    # Load predictions
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Load ground truth
    ground_truth = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data.get('document_id')
            if doc_id:
                # Convert annotations list to dict: {item: ground_truth}
                gt_dict = {ann['item']: ann['ground_truth'] for ann in data.get('annotations', [])}
                ground_truth[str(doc_id)] = gt_dict

    y_true = []
    y_pred = []
    
    total_items_evaluated = 0
    exact_matches = 0

    print(f"Evaluating {len(predictions)} patients...")

    for doc_id, pred_dict in predictions.items():
        if str(doc_id) not in ground_truth:
            continue
        
        gt_dict = ground_truth[str(doc_id)]
        
        for item, gt_val in gt_dict.items():
            # Get the predicted value, handling both nested dicts (new schema) and flat strings (old schema)
            pred_data = pred_dict.get(item, 'unknown')
            
            if isinstance(pred_data, dict):
                pred_val = str(pred_data.get('value', 'unknown')).strip().lower()
            else:
                pred_val = str(pred_data).strip().lower()
                
            gt_val_clean = str(gt_val).strip().lower()
            
            y_true.append(gt_val_clean)
            y_pred.append(pred_val)
            
            total_items_evaluated += 1
            if pred_val == gt_val_clean:
                exact_matches += 1

    if total_items_evaluated == 0:
        print("No matching items found to evaluate.")
        return

    accuracy = (exact_matches / total_items_evaluated) * 100
    
    print("\n" + "="*50)
    print(f"OVERALL EVALUATION RESULTS")
    print("="*50)
    print(f"Total Patients Evaluated: {len(predictions)}")
    print(f"Total CRF Items Evaluated: {total_items_evaluated}")
    print(f"Exact Match Accuracy: {accuracy:.2f}% ({exact_matches}/{total_items_evaluated})")
    print("="*50)
    
    # We will compute a confusion matrix specifically for the dominant categorical labels: y, n, unknown
    # Other specific values (like "97%" or "eupneic") will be grouped into an 'other' category for the matrix
    def categorize_label(lbl):
        if lbl in ['y', 'n', 'unknown']:
            return lbl
        return 'other'
        
    y_true_cat = [categorize_label(l) for l in y_true]
    y_pred_cat = [categorize_label(l) for l in y_pred]
    
    labels = ['y', 'n', 'unknown', 'other']
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    
    print("\nCONFUSION MATRIX (y / n / unknown / other)")
    print("-" * 50)
    # Print formatted confusion matrix
    header = f"{'':<12} | {'Pred: y':<10} | {'Pred: n':<10} | {'Pred: unk':<10} | {'Pred: other':<10}"
    print(header)
    print("-" * len(header))
    for i, true_label in enumerate(labels):
        row = f"True: {true_label:<7} | {cm[i][0]:<10} | {cm[i][1]:<10} | {cm[i][2]:<10} | {cm[i][3]:<10}"
        print(row)
        
    print("\nCLASSIFICATION REPORT")
    print("-" * 50)
    # Ignore warnings if some classes are never predicted
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(y_true_cat, y_pred_cat, labels=labels, output_dict=True)
        print(classification_report(y_true_cat, y_pred_cat, labels=labels))
        
        # Explicitly calculate F1(y+n)
        f1_y = report.get('y', {}).get('f1-score', 0)
        f1_n = report.get('n', {}).get('f1-score', 0)
        f1_yn_macro = (f1_y + f1_n) / 2
        
        print("\n" + "="*50)
        print(f"CRITICAL PAPER METRIC: Macro F1 (y+n) = {f1_yn_macro:.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRF JSON Predictions against Ground Truth")
    parser.add_argument("--preds", default="data/processed/predictions.json", help="Path to predictions JSON")
    parser.add_argument("--gt", default="data/raw/dev_gt.jsonl", help="Path to ground truth JSONL")
    args = parser.parse_args()
    
    evaluate_predictions(args.preds, args.gt)
