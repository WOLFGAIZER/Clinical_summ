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
You are a clinical reasoning and extraction system.

You are given:
1) A GLOBAL NARRATIVE (high-level patient trajectory)
2) VERIFIED CLINICAL EVENTS (ground truth with [S_xx])

-------------------------
GLOBAL NARRATIVE (CONTEXT ONLY):
{narrative}
-------------------------

-------------------------
VERIFIED CLINICAL EVENTS:
{context}
-------------------------

TASK:
For each CRF item, determine:
- "y" if condition is present
- "n" if condition is absent
- "unknown" if no information exists

-------------------------
DECISION RULES:

- Use VERIFIED EVENTS as the ONLY source of evidence
- Use GLOBAL NARRATIVE only to understand progression (not as evidence)
- Prefer "y" or "n" when reasonable evidence exists
- Use "unknown" ONLY if no relevant evidence is present
- Every "y" or "n" MUST include [S_xx]

-------------------------
CRF ITEMS:
{schema_list}

-------------------------
OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "item_name": "y | S_12",
  "item_name_2": "n | S_5",
  "item_name_3": "unknown"
}}

RULES:
- Output ONLY JSON
- No explanations
- No reasoning text
- No markdown
- No trailing commas
- End with }}
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