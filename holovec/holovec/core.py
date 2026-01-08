import numpy as np

class HyperVector:
    """
    The fundamental atom of HoloVec. 
    Represents a 10,000-dimensional bipolar vector {-1, 1}.
    """
    DIMENSIONS = 10000

    def __init__(self, data=None, seed=None):
        if data is not None:
            self.data = data
        else:
            # Random initialization: -1 or 1 with equal probability
            rng = np.random.default_rng(seed)
            self.data = rng.choice([-1, 1], size=self.DIMENSIONS).astype(np.int8)

    def __repr__(self):
        return f"<HyperVector dim={self.DIMENSIONS} hash={hash(self.data.tobytes())}>"

    def bundle(self, other):
        """
        The '+' Operator (Superposition).
        Combines two vectors. The result is similar to both inputs.
        Uses element-wise sum and normalization (majority rule).
        """
        # Element-wise addition
        res = self.data.astype(np.int32) + other.data.astype(np.int32)
        
        # Normalization (Sign function)
        # 0s are broken randomly to -1 or 1 to preserve information
        zero_indices = (res == 0)
        res[res > 0] = 1
        res[res < 0] = -1
        res[zero_indices] = np.random.choice([-1, 1], size=np.sum(zero_indices))
        
        return HyperVector(res.astype(np.int8))

    def bind(self, other):
        """
        The '*' Operator (Binding).
        Associates two vectors (e.g., Key * Value).
        The result is dissimilar to both inputs.
        Equivalent to XOR in boolean space.
        """
        return HyperVector(self.data * other.data)

    def permute(self, shifts=1):
        """
        The 'Π' Operator (Permutation).
        Encodes sequence/order by cyclically shifting the vector.
        """
        return HyperVector(np.roll(self.data, shifts))

    def similarity(self, other):
        """
        Cosine Similarity.
        Since vectors are bipolar (-1, 1), this is equivalent to 
        normalized Hamming distance.
        Returns value between -1.0 (opposite) and 1.0 (identical).
        """
        # Dot product / dimensions
        # Cast to int32 to prevent overflow during summation of 10k elements
        dot = np.dot(self.data.astype(np.int32), other.data.astype(np.int32))
        return dot / self.DIMENSIONS

    def __add__(self, other):
        return self.bundle(other)

    def __mul__(self, other):
        return self.bind(other)

    def invert(self):
        """
        The '-' Operator (Negation).
        Returns the inverse of the vector (flips -1 to 1 and vice versa).
        Useful for 'unlearning' or subtracting information.
        """
        return HyperVector(self.data * -1)

    def __neg__(self):
        return self.invert()

    @staticmethod
    def bundle_all(vectors):
        """
        Bundles a list of HyperVectors with EQUAL weight.
        Crucial for creating robust prototypes from multiple examples.
        """
        if not vectors:
            return HyperVector()
            
        # Accumulate raw sums
        accumulator = np.zeros(HyperVector.DIMENSIONS, dtype=np.int32)
        for vec in vectors:
            accumulator += vec.data.astype(np.int32)
            
        # Normalize
        zero_indices = (accumulator == 0)
        accumulator[accumulator > 0] = 1
        accumulator[accumulator < 0] = -1
        if np.any(zero_indices):
            accumulator[zero_indices] = np.random.choice([-1, 1], size=np.sum(zero_indices))
            
        return HyperVector(accumulator.astype(np.int8))
