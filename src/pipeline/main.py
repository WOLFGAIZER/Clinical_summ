from src.preprocess.wtts_builder import WTTSBuilder
from src.events.parser import parse_wtts
from src.embeddings.embedder import embed_event, embed_query, embed_events_batch
from src.retrieval.vector_store import EventVectorStore, rerank_with_weights
from src.generation.generator import generate_crf, generate_patient_narrative
from src.generation.query_generator import generate_queries

import json
import os
import concurrent.futures
import re


# -------------------------------
# Compression
# -------------------------------
def compress_wtts(wtts_string, max_events=30):
    parts = wtts_string.split("|")
    return " | ".join(parts[-max_events:])

def get_narrative_cache_path(pid):
    return f"data/processed/narratives/{pid}.txt"

def safe_json_load(content):
    # Extract JSON block
    start = content.find('{')
    end = content.rfind('}')

    if start == -1 or end == -1:
        raise ValueError("No JSON found")

    json_str = content[start:end+1]

    # Fix common issues
    json_str = re.sub(r",\s*}", "}", json_str)  # trailing commas
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except:
        # Last fallback: replace single quotes
        json_str = json_str.replace("'", '"')
        return json.loads(json_str)

# -------------------------------
# Utility: Load CRF schema keys
# -------------------------------
def get_crf_schema_keys(gt_path="data/raw/dev_gt.jsonl"):
    keys = set()
    valid_patient_ids = set()

    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'document_id' in data:
                    valid_patient_ids.add(str(data['document_id']))

                for ann in data.get('annotations', []):
                    keys.add(ann['item'])
    except Exception as e:
        print(f"[WARNING] Failed loading GT: {e}")

    return sorted(list(keys)), valid_patient_ids


# -------------------------------
def expand_temporal(events, full_event_list):
    expanded = []

    for e in events:
        idx = full_event_list.index(e)

        for i in [idx-1, idx, idx+1]:
            if 0 <= i < len(full_event_list):
                expanded.append(full_event_list[i])

    return list(set(expanded))

# -------------------------------
# Core Pipeline
# -------------------------------
def run_pipeline(patient_data, crf_schema_keys):

    pid = str(
        patient_data.get('document_id')
        or patient_data.get('patient_id')
        or "unknown_id"
    )

    print(f"\n[START] Patient: {pid}")

    builder = WTTSBuilder()

    # -------------------------------
    # Step 1: WTTS (cache-safe)
    # -------------------------------
    wtts_path = f"data/processed/materialized_ehr/{pid}.jsonl"
    wtts_string = ""

    if os.path.exists(wtts_path):
        try:
            print("[CACHE] Loading WTTS...")
            with open(wtts_path, 'r') as f:
                data = json.load(f)

            wtts_string = data.get("text", "")

            if not wtts_string or len(wtts_string) < 10:
                raise ValueError("Corrupt WTTS")

        except Exception as e:
            print(f"[CACHE ERROR] {e} -> regenerating")
            wtts_string = builder.build_wtts_string(patient_data)

            with open(wtts_path, 'w') as f:
                json.dump({"person_id": pid, "text": wtts_string}, f)

    else:
        print("[WTTS] Generating...")
        wtts_string = builder.build_wtts_string(patient_data)

        os.makedirs("data/processed/materialized_ehr", exist_ok=True)
        with open(wtts_path, 'w') as f:
            json.dump({"person_id": pid, "text": wtts_string}, f)

    # -------------------------------
    # Step 1.5: Narrative (cached)
    # -------------------------------
    narrative_dir = "data/processed/narratives"
    os.makedirs(narrative_dir, exist_ok=True)

    narrative_path = get_narrative_cache_path(pid)

    if os.path.exists(narrative_path):
        print("[CACHE] Loading Narrative...")
        with open(narrative_path, 'r', encoding='utf-8') as f:
            narrative = f.read()
    else:
        print("[NARRATIVE] Generating...")

        # Optional compression for better signal
        short_wtts = compress_wtts(wtts_string)

        narrative = generate_patient_narrative(short_wtts)

        with open(narrative_path, 'w', encoding='utf-8') as f:
            f.write(narrative)

    # -------------------------------
    # Step 2: Parse events
    # -------------------------------
    events = parse_wtts(wtts_string)

    if not events:
        print(f"[SKIP] No events for {pid}")
        return {}

    # -------------------------------
    # Step 3: Filter useless events
    # -------------------------------
    original_events = events.copy()

    # Apply filtering only if large set
    if len(events) > 20:
        events = [e for e in events if e[3] > 0.05]

    # Restore if too few
    if len(events) < 10:
        print("[INFO] Too few events -> restoring original events")
        events = original_events

    if not events:
        print(f"[SKIP] All events filtered out for {pid}")
        return {}

    print(f"[INFO] Events after filtering: {len(events)}")

    # -------------------------------
    # Step 4: Build vector store
    # -------------------------------
    store = EventVectorStore()
    embeddings = embed_events_batch(events)

    for emb, e in zip(embeddings, events):
        store.add(emb, e)

    # -------------------------------
    # Step 5: CRF-Aware Retrieval
    # -------------------------------
    pooled_candidates = {}

    for crf_item in crf_schema_keys:
        q_emb = embed_query(crf_item)

        retrieved = store.search(q_emb, k=2)

        scored = rerank_with_weights(
            retrieved_results=retrieved,
            target_time=0.8,
            alpha=0.6,
            beta=0.3,
            gamma=0.3
        )

        for score, event in scored:
            if len(event) < 5:
                continue

            sid = event[4]

            if sid not in pooled_candidates or score > pooled_candidates[sid][0]:
                pooled_candidates[sid] = (score, event)

    if not pooled_candidates:
        print(f"[SKIP] No retrieved candidates for {pid}")
        return {}

    print(f"[INFO] Candidates pooled: {len(pooled_candidates)}")

    # -------------------------------
    # Step 7: Select top events
    # -------------------------------
    filtered_candidates = list(pooled_candidates.values())

    # We now keep ALL pooled candidates for the CRF items instead of globally slicing to [:12]
    # This fixes the Evidence Starvation bug!
    final_events = [e for _, e in filtered_candidates]

    if not final_events:
        print(f"[SKIP] No meaningful events after ranking for {pid}")
        return {}

    # Expand temporally to grab neighbors (n-1, n, n+1)
    final_events = expand_temporal(final_events, original_events)

    # SOFT CAP
    MAX_EVENTS = 60
    if len(final_events) > MAX_EVENTS:
        final_events = sorted(
            final_events,
            key=lambda x: x[3],  # sort by weight W
            reverse=True
        )[:MAX_EVENTS]

    # -------------------------------
    # Step 8: Temporal ordering
    # -------------------------------
    final_events_sorted = sorted(final_events, key=lambda x: x[2])

    # -------------------------------
    # Step 9: Tier 3 Fusion CoT
    # -------------------------------
    try:
        output = generate_crf(final_events_sorted, crf_schema_keys, narrative)
    except Exception as e:
        print(f"[LLM ERROR] {pid}: {e}")
        return {}

    print(f"[DONE] Patient: {pid}")

    return output


# -------------------------------
# Post-processing
# -------------------------------
def process_patient(patient_data, crf_schema_keys, idx, total):

    print(f"\n{'='*50}")
    print(f"Patient {idx+1}/{total} -> {patient_data['document_id']}")
    print(f"{'='*50}")

    try:
        result_json = run_pipeline(patient_data, crf_schema_keys)

        if not result_json:
            return patient_data['document_id'], None

        parsed = safe_json_load(result_json)

        final_output = {}

        for key, val in parsed.items():

            if not isinstance(val, str):
                continue

            if "|" not in val:
                continue

            parts = val.split("|", 1)

            value = parts[0].strip().lower()

            evidence = []
            if len(parts) > 1:
                evidence = [e.strip() for e in parts[1].split(",") if e.strip()]

            if value == "unknown" or not evidence:
                continue

            final_output[key] = {
                "value": value,
                "evidence": evidence
            }

        return patient_data['document_id'], final_output

    except Exception as e:
        print(f"[ERROR] {patient_data['document_id']}: {e}")
        return patient_data['document_id'], None


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":

    from src.preprocess.data_loader import DataLoader

    input_dirs = [
        "data/raw/dyspnea-clinical-notes",
        "data/raw/dyspnea-crf-development"
    ]

    print("[LOAD] Loading datasets...")
    loader = DataLoader(data_folders=input_dirs)
    patients = loader.load_and_merge()

    crf_schema_keys, valid_ids = get_crf_schema_keys()

    eval_patients = [
        p for p in patients
        if str(p.get('document_id')) in valid_ids
    ]

    print(f"[INFO] Valid patients: {len(eval_patients)}")

    results = {}

    test_batch_size = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

        futures = [
            executor.submit(process_patient, p, crf_schema_keys, i, test_batch_size)
            for i, p in enumerate(eval_patients[:test_batch_size])
        ]

        for future in concurrent.futures.as_completed(futures):
            doc_id, parsed = future.result()
            if parsed:
                results[doc_id] = parsed

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/predictions.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n[SUCCESS] Predictions saved.")