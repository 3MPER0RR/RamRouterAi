import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EpisodicMemory:

    def __init__(self):
        self.memories = []

    def add(self, text, embedding, response=None, score=0.0):

        self.memories.append({
            "timestamp": time.time(),
            "text": text,
            "response": response,
            "score": score,
            "embedding": embedding
        })

    def retrieve(self, embedding, top_k=3):

        if not self.memories:
            return []

        matrix = np.array([
            m["embedding"] for m in self.memories
        ])

        sims = cosine_similarity(
            [embedding],
            matrix
        )[0]

        idx = sims.argsort()[::-1][:top_k]

        return [
            self.memories[i]
            for i in idx
        ]
