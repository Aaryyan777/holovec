import numpy as np
from .core import HyperVector

class AssociativeMemory:
    """
    A 'Brain' that stores concepts and allows fuzzy retrieval.
    """
    def __init__(self):
        self.memory = {}

    def add(self, label, vector):
        """
        Learns a concept. If the label exists, it updates the concept
        by bundling the new observation (Reinforcement Learning).
        """
        if label in self.memory:
            self.memory[label] = self.memory[label].bundle(vector)
        else:
            self.memory[label] = vector

    def query(self, query_vector, top_k=1):
        """
        Finds the concept in memory most similar to the query vector.
        """
        results = []
        for label, stored_vector in self.memory.items():
            score = query_vector.similarity(stored_vector)
            results.append((label, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
