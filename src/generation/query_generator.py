import ollama
import json

def generate_queries(patient_data):
    """
    Generate dynamic queries for retrieval based on patient data.
    The output strictly formats into respiratory, cardiac, and infection queries.
    """
    
    # Simple summary of patient admission for context
    notes_summary = ""
    if "notes" in patient_data and len(patient_data["notes"]) > 0:
        notes_summary = patient_data["notes"][0]["text"]
        
    prompt = f"""
You are an expert clinical assistant.
Based on the following admission note, generate three distinct short search queries to retrieve relevant clinical events from the patient's timeline.
Focus specifically on detecting deterioration in these three areas:
1. Respiratory
2. Cardiac
3. Infection

Admission Note:
{notes_summary}

Output ONLY a valid JSON array of 3 strings, with no additional text or markdown.
Example format:
[
  "respiratory deterioration hypoxia",
  "cardiac deterioration hemodynamic instability",
  "infection sepsis fever"
]
"""

    try:
        response = ollama.chat(
            model='llama3:8b',
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()
        
        # Strip markdown json block if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        queries = json.loads(content.strip())
        if isinstance(queries, list) and len(queries) > 0:
            return queries
    except Exception as e:
        print("[WARN] Query generation failed → using fallback")
        
        return [
            "respiratory symptoms dyspnea shortness of breath",
            "cardiac instability hypotension shock heart rate",
            "infection fever sepsis wbc antibiotics"
        ]
