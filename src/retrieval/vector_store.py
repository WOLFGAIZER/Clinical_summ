import faiss
import numpy as np


class EventVectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.events = []

    def add(self, embedding, event):
        """
        event = (event_text, timestamp, P_j, W, sentence_id)
        """
        self.index.add(np.array([embedding]).astype("float32"))
        self.events.append(event)

    def search(self, query_embedding, k=5):
        """
        Returns:
        [
            (similarity_score, event_tuple),
            ...
        ]
        """

        query_embedding = np.array([query_embedding]).astype("float32")

        D, I = self.index.search(query_embedding, k)

        results = []

        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.events):
                continue

            event = self.events[idx]

            # Convert L2 distance → similarity
            similarity = 1 / (1 + dist)

            results.append((similarity, event))

        return results


# -------------------------------
# NEW: Re-ranking with clinical logic
# -------------------------------

def rerank_with_weights(retrieved_results, target_time=0.8, alpha=0.6, beta=0.3, gamma=0.1):
    """
    retrieved_results:
        [(similarity, (event, timestamp, P_j, W, sentence_id)), ...]

    target_time:
        normalized time (0 → admission, 1 → discharge)

    Returns:
        [(final_score, event_tuple), ...]
    """

    scored = []

    for similarity, event in retrieved_results:
        event_text, timestamp, P_j, W, sentence_id = event

        # Temporal relevance (closer to target_time is better)
        temporal_score = 1 - abs(P_j - target_time)

        # Final scoring function
        final_score = (
            alpha * similarity +
            beta * W +
            gamma * temporal_score
        )

        scored.append((final_score, event))

    return scored