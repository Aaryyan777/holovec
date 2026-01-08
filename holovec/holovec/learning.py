import numpy as np
from .core import HyperVector

class PerceptronClassifier:
    """
    An iterative learner for Hyperdimensional Computing.
    Uses the 'Perceptron' algorithm to fine-tune class prototypes.
    
    Novelty:
    - Starts with One-Shot learning (Instant).
    - Improves iteratively (High Accuracy).
    - Uses Integer Hypervectors for precision during training.
    """
    def __init__(self, dimensions=10000):
        self.dimensions = dimensions
        # We store 'raw' integer sums for training precision
        # { "label": np.array([5, -2, 10, ...]) }
        self.prototypes = {}

    def fit(self, X_train, y_train, epochs=5, learning_rate=1.0):
        """
        Train the model iteratively.
        """
        # 1. Initialization (One-Shot Phase)
        print("Phase 1: Initialization (One-Shot)...")
        for vector, label in zip(X_train, y_train):
            if label not in self.prototypes:
                self.prototypes[label] = np.zeros(self.dimensions, dtype=np.int32)
            self.prototypes[label] += vector.data.astype(np.int32)
            
        # 2. Iterative Refinement (Perceptron Phase)
        print(f"Phase 2: Iterative Refinement ({epochs} epochs)...")
        for epoch in range(epochs):
            errors = 0
            for vector, label in zip(X_train, y_train):
                # Predict
                predicted_label = self.predict_one(vector)
                
                # Update if wrong
                if predicted_label != label:
                    errors += 1
                    # Perceptron Rule:
                    # Move Correct Class TOWARDS input
                    self.prototypes[label] += (vector.data.astype(np.int32) * int(learning_rate))
                    # Move Wrong Class AWAY from input
                    self.prototypes[predicted_label] -= (vector.data.astype(np.int32) * int(learning_rate))
            
            acc = 1.0 - (errors / len(X_train))
            print(f"  Epoch {epoch+1}/{epochs} - Accuracy: {acc:.4f}")
            if errors == 0:
                print("  Converged early!")
                break

    def predict_one(self, vector):
        """
        Predicts the class for a single HyperVector.
        """
        best_label = None
        best_sim = -float('inf')
        
        # We simulate dot product against the integer prototypes
        # This is equivalent to Cosine Similarity
        vec_data = vector.data.astype(np.int32)
        
        for label, proto in self.prototypes.items():
            # Dot product
            sim = np.dot(proto, vec_data)
            if sim > best_sim:
                best_sim = sim
                best_label = label
                
        return best_label

    def predict(self, X_test):
        return [self.predict_one(vec) for vec in X_test]
