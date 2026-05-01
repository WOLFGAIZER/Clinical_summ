import ollama


import json

def generate_crf(events, crf_schema_keys, narrative):
    """
    events:
    [
        (event, timestamp, P_j, W, sentence_id),
        ...
    ]
    """

    # Build context
    context_lines = []
    for event, timestamp, P_j, W, sid in events:
        context_lines.append(
            f"[{sid}] {timestamp} | {event} | importance:{W} | time:{P_j}"
        )
    context = "\n".join(context_lines)

    schema_list = "\n".join([f"- {k}" for k in crf_schema_keys])

    prompt = f"""
-------------------------
GLOBAL NARRATIVE (CONTEXT ONLY):
{narrative}
-------------------------

-------------------------
VERIFIED CLINICAL EVENTS (GROUND TRUTH):
{context}
-------------------------

INSTRUCTIONS:

Step 1: Identify abnormal events using VERIFIED EVENTS ONLY
Step 2: Use GLOBAL NARRATIVE only to understand sequence
Step 3: Link events across time using VERIFIED EVENTS
Step 4: Identify key clinical conditions
Step 5: Extract CRF values using ONLY VERIFIED EVENTS

STRICT RULES:
- Narrative MUST NOT be used as evidence
- Every answer MUST include an [S_xx] from VERIFIED EVENTS
- Output MUST be ONLY valid JSON (no markdown)

Use "unknown" ONLY if:
- No relevant event exists
- AND no reasonable inference can be made

If evidence suggests absence:
-> return "n | S_xx"

If evidence suggests presence:
-> return "y | S_xx"

EXTRACTION GUIDELINES:
- The system prefers confident decisions over conservative defaults.
- Prefer "y" or "n" when reasonable evidence exists
- Do NOT default to "unknown"
- BUT do NOT guess without evidence

-------------------------
CRF ITEMS:
{schema_list}

OUTPUT FORMAT:
{{
  "item_name": "value | S_xx"
}}

IMPORTANT:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include trailing commas
- Every key MUST be in double quotes
- End output with }}
"""

    resp = ollama.chat(
        model='llama3:8b',
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp['message']['content'].strip()

    # Robust JSON extraction
    start = content.find('{')
    end = content.rfind('}')
    
    if start != -1:
        if end != -1 and end >= start:
            content = content[start:end+1]
        else:
            # Output was likely truncated
            content = content[start:] + '\n}'

    # JSON Repair Layer
    import re
    # Remove markdown formatting
    content = re.sub(r'```json|```', '', content).strip()
    # Remove trailing commas before a closing brace
    content = re.sub(r',\s*}', '}', content)
    # Ensure it closes
    if not content.endswith('}'):
        content += '\n}'

    return content.strip()

def generate_patient_narrative(wtts_string: str) -> str:
    """
    Generates a grounded, short narrative from WTTS.
    """
    prompt = f"""
You are a clinical summarizer.

STRICT RULES:
- Use ONLY the provided WTTS events
- DO NOT infer missing conditions
- DO NOT hallucinate causal links
- Keep it concise (4–6 sentences)

TASK:
Write a short chronological summary focusing on:
- major events
- progression over time
- observed cause-effect ONLY if explicitly present

WTTS:
{wtts_string}

Output:
Clinical Narrative:
"""

    resp = ollama.chat(
        model='llama3:8b',
        messages=[{"role": "user", "content": prompt}]
    )
    return resp['message']['content'].strip()