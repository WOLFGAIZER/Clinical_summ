import re

def parse_wtts(wtts_string):
    """
    Convert WTTS string into structured tuples:
    (event, timestamp, P_j, W, sentence_id)
    """

    events = []

    # Split by top-level separator
    parts = wtts_string.split("|")

    for p in parts:
        try:
            text = p.strip()

            # Extract sentence ID → [S_12]
            sid_match = re.search(r'\[(.*?)\]', text)
            sentence_id = sid_match.group(1) if sid_match else None

            # Extract inside tuple: ("date", "event", P_j, W)
            tuple_match = re.search(
                r'\("([^"]+)",\s*"([^"]+)",\s*([\d\.]+),\s*([\d\.]+)\)',
                text
            )

            if not tuple_match:
                continue

            timestamp = tuple_match.group(1)
            event = tuple_match.group(2)
            P_j = float(tuple_match.group(3))
            W = float(tuple_match.group(4))

            events.append((event, timestamp, P_j, W, sentence_id))

        except Exception:
            continue

    return events