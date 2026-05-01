from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en')

_EMBEDDING_CACHE = {}

def embed_event(event_tuple):
    """
    event_tuple = (event, timestamp, P_j, W, sentence_id)
    """

    event, timestamp, P_j, W, sentence_id = event_tuple

    # Build richer semantic + structured representation
    text = (
        f"{event}. "
        f"Clinical importance level: {W}. "
        f"Timeline position: {P_j}."
    )

    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]
        
    emb = model.encode(text)
    _EMBEDDING_CACHE[text] = emb
    return emb

def embed_events_batch(event_tuples):
    """
    Batch process events for faster inference using cache.
    """
    texts = []
    for event, timestamp, P_j, W, sentence_id in event_tuples:
        texts.append(
            f"{event}. "
            f"Clinical importance level: {W}. "
            f"Timeline position: {P_j}."
        )
        
    embeddings = [None] * len(texts)
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        if text in _EMBEDDING_CACHE:
            embeddings[i] = _EMBEDDING_CACHE[text]
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
            
    if uncached_texts:
        new_embs = model.encode(uncached_texts, batch_size=32)
        for i, emb in zip(uncached_indices, new_embs):
            embeddings[i] = emb
            _EMBEDDING_CACHE[texts[i]] = emb
            
    return embeddings


def embed_query(query):
    """
    Query embedding remains simple but can be extended later
    """
    if query in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[query]
        
    emb = model.encode(query)
    _EMBEDDING_CACHE[query] = emb
    return emb