import os
import glob
import pandas as pd

class DataLoader:
    def __init__(self, data_folders, gt_path=None):
        self.data_folders = data_folders
        self.gt_path = gt_path

    def load_and_merge(self):
        """
        Recursively searches for .parquet files in the provided data folders.
        Loads them and formats them into patient_data dictionaries.
        """
        all_patients = []
        
        for folder in self.data_folders:
            # Search for parquet files recursively
            search_pattern = os.path.join(folder, '**', '*.parquet')
            parquet_files = glob.glob(search_pattern, recursive=True)
            
            for pf in parquet_files:
                # We prioritize English files if split by language, but load all found otherwise
                # (The C4Health dataset has en-* and it-* files)
                if 'it-' in os.path.basename(pf):
                    continue  # Skip Italian files for now
                    
                print(f"DataLoader: Reading {pf}")
                try:
                    df = pd.read_parquet(pf)
                    for _, row in df.iterrows():
                        # Extract required fields and strip language suffixes to match ground truth IDs
                        doc_id = str(row.get('document_id', 'unknown'))
                        if doc_id.endswith('_en'): doc_id = doc_id[:-3]
                        if doc_id.endswith('_it'): doc_id = doc_id[:-3]
                        
                        note_text = str(row.get('clinical_note', ''))
                        
                        # Format as expected by wtts_builder and the pipeline
                        # Since C4Health is a single note with no timestamps, we mock the timeline
                        patient_data = {
                            "document_id": doc_id,
                            "admission_time": "2026-01-01",  # Mocked start date
                            "discharge_time": "2026-01-10",  # Mocked end date
                            "notes": [
                                {
                                    "timestamp": "2026-01-01", # Mocked note date
                                    "text": note_text
                                }
                            ]
                        }
                        all_patients.append(patient_data)
                except Exception as e:
                    print(f"Error reading {pf}: {e}")
                    
        print(f"DataLoader: Successfully loaded {len(all_patients)} total patients.")
        return all_patients
