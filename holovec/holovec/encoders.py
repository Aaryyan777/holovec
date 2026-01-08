from .core import HyperVector
import string
import numpy as np

class TextEncoder:
    """
    Encodes text into HyperVectors using N-gram statistics.
    This preserves local context and sequence information.
    """
    def __init__(self, ngram_size=3):
        self.ngram_size = ngram_size
        self.item_memory = {} # Static vectors for each character
        self._init_memory()

    def _init_memory(self):
        # Create a random unique vector for every printable character
        chars = string.printable
        for char in chars:
            self.item_memory[char] = HyperVector()
        # Fallback for unknown chars
        self.item_memory['<UNK>'] = HyperVector()

    def _get_vector(self, char):
        return self.item_memory.get(char, self.item_memory['<UNK>'])

    def encode(self, text):
        """
        Encodes a string into a single HyperVector.
        Method: Sum of N-grams.
        Example (3-gram): "HELLO" -> bind(H, E, L) + bind(E, L, L) + ...
        """
        if len(text) < self.ngram_size:
            text = text.ljust(self.ngram_size) # Pad if too short

        # Accumulator for the raw sum (int32 to prevent overflow)
        # We access the DIMENSIONS from the class or an instance
        accumulator = np.zeros(HyperVector.DIMENSIONS, dtype=np.int32)

        # Sliding window
        for i in range(len(text) - self.ngram_size + 1):
            ngram_window = text[i : i + self.ngram_size]
            
            # Create N-gram vector
            ngram_vector = self._get_vector(ngram_window[0])
            for shift, char in enumerate(ngram_window[1:], start=1):
                # Bind rotated vectors: V[0] * Rot(V[1]) * Rot(Rot(V[2]))...
                char_vec = self._get_vector(char)
                ngram_vector = ngram_vector.bind(char_vec.permute(shift))

            # Accumulate raw values (no normalization yet)
            accumulator += ngram_vector.data.astype(np.int32)
            
        # Final Normalization (Sign function)
        # 0s are broken randomly to -1 or 1
        zero_indices = (accumulator == 0)
        accumulator[accumulator > 0] = 1
        accumulator[accumulator < 0] = -1
        # Randomly resolve zeros if any
        if np.any(zero_indices):
             # We need a seed or random generator. Using numpy's default.
            accumulator[zero_indices] = np.random.choice([-1, 1], size=np.sum(zero_indices))

        return HyperVector(accumulator.astype(np.int8))

class ProjectionEncoder:
    """
    Encodes numerical vectors (like images) into HyperVectors.
    Uses Random Projection: Sign(Input @ Matrix).
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = HyperVector.DIMENSIONS
        # Create a random projection matrix (Input Dim x 10,000)
        # Values are drawn from a normal distribution
        rng = np.random.default_rng(42) # Fixed seed for reproducibility
        self.projection_matrix = rng.normal(0, 1, size=(self.input_dim, self.output_dim))

    def encode(self, input_vector):
        """
        Encodes a numpy array of shape (input_dim,) into a HyperVector.
        """
        # Ensure input is 1D
        input_vector = np.array(input_vector).flatten()
        if input_vector.shape[0] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {input_vector.shape[0]}")
            
        # Projection: Input . Matrix
        # This mixes every input feature into every output dimension
        projected = np.dot(input_vector, self.projection_matrix)
        
        # Binarize (Sign function)
        # -1 if x < 0, 1 if x > 0
        # We handle 0s by mapping them to 1 for simplicity (or random)
        binary_data = np.where(projected >= 0, 1, -1).astype(np.int8)
        
        return HyperVector(binary_data)
        
    def encode_batch(self, batch_matrix):
        """
        Encodes a batch of inputs (N x Input_Dim) -> List[HyperVector]
        Optimization: Uses matrix multiplication for the whole batch.
        """
        # Batch Projection
        projected = np.dot(batch_matrix, self.projection_matrix)
        
        # Binarize
        binary_batch = np.where(projected >= 0, 1, -1).astype(np.int8)
        
        # Convert to HyperVector objects
        return [HyperVector(row) for row in binary_batch]
