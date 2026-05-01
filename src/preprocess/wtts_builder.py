import re
import json
import os
import pandas as pd
import argparse
from datetime import datetime
from src.preprocess.data_loader import DataLoader

class WTTSBuilder:
    def __init__(self):
        # --- WEIGHTING RULES (W) ---
        self.critical_patterns = [
            r'respiratory failure', r'seizure', r'cardiac arrest', r'intubat', 
            r'abnormal', r'critical', r'hemorrhage', r'positive', 
            r'emergency', r'acute', r'hypoxia', r'flagged', r'icu',
            r'dyspnea', r'shortness of breath', r'sob', r'mrc grade', r'nyha'
        ]
        self.chronic_patterns = [
            r'history of', r'chronic', r'stable', r'continued', 
            r'maintained', r'diagnosed with', r'previous', r'known'
        ]
        self.routine_patterns = [
            r'routine', r'normal', r'negative', r'unremarkable', 
            r'no acute', r'clear', r'regular diet', r'resting'
        ]

    def _get_normalized_time(self, event_time_str, admit_str, disch_str):
        """Calculates P_j (0.0 to 1.0)"""
        try:
            e_dt = pd.to_datetime(event_time_str)
            a_dt = pd.to_datetime(admit_str)
            d_dt = pd.to_datetime(disch_str)
            
            total_duration = (d_dt - a_dt).total_seconds()
            elapsed = (e_dt - a_dt).total_seconds()
            
            if total_duration <= 0: return 1.0
            return round(max(0.0, min(1.0, elapsed / total_duration)), 2)
        except:
            return 0.5

    def _get_weight(self, text):
        t = text.lower()
        if any(re.search(p, t) for p in self.critical_patterns): return 1.0
        if any(re.search(p, t) for p in self.chronic_patterns): return 0.5
        if any(re.search(p, t) for p in self.routine_patterns): return 0.1
        return 0.5 

    def _is_junk(self, text):
        t = text.lower()
        junk_patterns = [
            r'electronically signed by', r'md\s*$', r'm\.d\.\s*$', r'dob:', r'date of birth', 
            r'admission date:', r'discharge date:', r'dictated by:', r'attending:',
            r'job id:', r'^page \d+ of \d+', r'^\s*$', r'confidential'
        ]
        return any(re.search(p, t) for p in junk_patterns)

    def _extract_sentences_with_ids(self, text, start_index):
        """
        Splits notes and assigns UNIQUE IDs.
        CRITICAL FIX: Sanitizes newlines to preserve WTTS structure.
        """
        # 1. Replace newlines/tabs with spaces to keep tuple on one line
        text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        
        # 2. Remove de-id brackets
        text = re.sub(r'\[\*\*.*?\*\*\]', '', text) 
        
        # 3. Split by sentence boundaries
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        results = []
        current_idx = start_index
        for s in sentences:
            clean_s = s.strip()
            # Ignore very short/empty fragments and junk data
            if len(clean_s) > 5 and not self._is_junk(clean_s):
                sid = f"S_{current_idx}"
                results.append((sid, clean_s))
                current_idx += 1
        return results, current_idx

    def _clean_text(self, text):
        if not text:
            return ""
        text = text.replace('"', "'")          # avoid JSON break
        text = re.sub(r'\s+', ' ', text)      # collapse spaces
        text = re.sub(r'[\x00-\x1f]', '', text)  # remove control chars
        return text.strip()

    def build_wtts_string(self, patient_data):
        tuples = []
        admit = patient_data.get('admission_time')
        disch = patient_data.get('discharge_time')
        
        sorted_notes = sorted(patient_data.get('notes', []), key=lambda x: x['timestamp'])
        sorted_notes = sorted_notes[-40:]  # Limit notes to last 40
        
        global_sent_idx = 0 

        for note in sorted_notes:
            raw_ts = note['timestamp']
            p_j = self._get_normalized_time(raw_ts, admit, disch)
            
            cleaned_text = self._clean_text(note['text'])
            events, global_sent_idx = self._extract_sentences_with_ids(cleaned_text, global_sent_idx)
            
            for (sid, event) in events:
                w = self._get_weight(event)
                # Format: [ID] ("Date", "Event", P_j, W)
                tuples.append(f'[{sid}] ("{raw_ts}", "{event}", {p_j}, {w})')
        
        return " | ".join(tuples)

# --- EXECUTION LOGIC ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw clinical notes into WTTS format.")
    
    # Dataset Citation: C4Health (CL4Health2026) Dataset
    # Raw notes: data/raw/dyspnea-clinical-notes
    # CRF Development data: data/raw/dyspnea-crf-development
    # Ground truth: data/raw/dev_gt.jsonl
    
    parser.add_argument("--input_dirs", nargs="+",
                        default=[
                            "data/raw/dyspnea-clinical-notes",
                            "data/raw/dyspnea-crf-development",
                        ],
                        help="Directories containing .parquet shards (searched recursively)")
    parser.add_argument("--gt_file", type=str, 
                        default="data/raw/dev_gt.jsonl",
                        help="Path to the ground truth JSONL file.")
    parser.add_argument("--output_dir", type=str, 
                        default="data/processed/materialized_ehr",
                        help="Path to store processed JSONL files.")
    
    args = parser.parse_args()

    loader = DataLoader(data_folders=args.input_dirs, gt_path=args.gt_file)
    builder = WTTSBuilder()

    patients = loader.load_and_merge()
    os.makedirs(args.output_dir, exist_ok=True)

    if not patients:
        print("No patients found! Check paths.")
    else:
        print(f"Materializing timelines for {len(patients)} patients...")
        for p in patients:
            wtts_output = builder.build_wtts_string(p)
            
            # FIX: Prioritize document_id to match DataLoader logic
            pid = str(p.get('document_id') or p.get('patient_id') or p.get('hadm_id'))
            
            output_path = os.path.join(args.output_dir, f"{pid}.jsonl")
            with open(output_path, 'w') as f:
                json.dump({"person_id": pid, "text": wtts_output}, f)
        
        print(f"Successfully stored outputs in: {args.output_dir}")