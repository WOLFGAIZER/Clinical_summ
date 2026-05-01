import sys
from unittest.mock import patch
import src.pipeline.main as main_module

def mock_chat(*args, **kwargs):
    if "generate three distinct short search queries" in kwargs.get('messages', [{}])[0].get('content', ''):
        return {'message': {'content': '["respiratory deterioration hypoxia", "cardiac deterioration", "infection fever"]'}}
    else:
        return {'message': {'content': '{"history of allergy": "n", "chest pain": "unknown", "spo2": "98%"}'}}

with patch('ollama.chat', side_effect=mock_chat):
    try:
        print("Running main.py block with mocked LLM to verify pipeline...")
        
        input_dirs = [
            "data/raw/dyspnea-clinical-notes",
            "data/raw/dyspnea-crf-development"
        ]
        from src.preprocess.data_loader import DataLoader
        loader = DataLoader(data_folders=input_dirs)
        patients = loader.load_and_merge()
        
        if not patients:
            print("No patients found.")
            sys.exit(1)
            
        test_batch_size = 2 # Verify on 2 patients for speed
        crf_schema_keys = main_module.get_crf_schema_keys()
        for i, patient_data in enumerate(patients[:test_batch_size]):
            print(f"\nVerifying Patient {patient_data['document_id']}")
            result = main_module.run_pipeline(patient_data, crf_schema_keys)
            print(result)
            
        print("\nVerification successful! Pipeline completed without breaks.")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
