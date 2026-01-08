import holovec as hv
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def main():
    print("=== HoloVec: Fashion-MNIST Benchmark ===")
    print("Task: Classify 28x28 grayscale images of clothing (10 classes).")
    print("Strategy: Random Projection -> Hyperdimensional Computing.")
    print("-" * 50)

import urllib.request
import gzip
import os

def load_fashion_mnist_manual():
    """
    Manually downloads and parses Fashion-MNIST training data.
    """
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        'X': 'train-images-idx3-ubyte.gz',
        'y': 'train-labels-idx1-ubyte.gz'
    }
    
    data = {}
    
    for key, filename in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filename)
            
        with gzip.open(filename, 'rb') as f:
            if key == 'X':
                # Skip magic number (4) and dims (4*3 = 12) -> 16 bytes
                f.read(16)
                # Read rest
                buf = f.read()
                data[key] = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 784)
            else:
                # Skip magic number (4) and count (4) -> 8 bytes
                f.read(8)
                buf = f.read()
                data[key] = np.frombuffer(buf, dtype=np.uint8)
                
    return data['X'], data['y']

def main():
    print("=== HoloVec: Fashion-MNIST Benchmark ===")
    print("Task: Classify 28x28 grayscale images of clothing (10 classes).")
    print("Strategy: Random Projection -> Hyperdimensional Computing.")
    print("-" * 50)

    # 1. Load Data
    print("Loading Fashion-MNIST (Manual Download)...")
    try:
        X, y = load_fashion_mnist_manual()
        # Convert labels to strings because HoloVec expects hashable labels (ints work too, but let's be safe)
        y = y.astype(str)
    except Exception as e:
        print(f"Error downloading/parsing dataset: {e}")
        return
    
    # Normalize pixel values to 0-1 range
    X = X / 255.0
    
    # Subset for speed (4000 train, 500 test)
    # HDC is fast, but Random Projection of 4000 images x (784*10000) ops is heavy in Python
    TRAIN_SIZE = 4000
    TEST_SIZE = 500
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    
    print(f"Data Loaded. Training on {TRAIN_SIZE} images, Testing on {TEST_SIZE}.")
    
    # 2. Encode Images (The "Novel" Part)
    print("\n[Step 1] Encoding Images to 10,000-D HyperVectors...")
    t0 = time.time()
    
    # Initialize Projection Encoder for 28x28 = 784 inputs
    encoder = hv.ProjectionEncoder(input_dim=784)
    
    # Encode batches
    # We do this once to save time for both experiments
    print("  Encoding Training Set...")
    train_vectors = encoder.encode_batch(X_train)
    
    print("  Encoding Test Set...")
    test_vectors = encoder.encode_batch(X_test)
    
    print(f"Encoding finished in {time.time() - t0:.2f} seconds.")

    # 3. Experiment A: One-Shot Learning
    print("\n[Experiment A] One-Shot Learning (Single Pass)")
    t0 = time.time()
    
    memory = hv.AssociativeMemory()
    
    # Group by class
    class_data = {}
    for vec, label in zip(train_vectors, y_train):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(vec)
        
    # Bundle
    for label, vecs in class_data.items():
        proto = hv.HyperVector.bundle_all(vecs)
        memory.add(label, proto)
        
    # Evaluate
    predictions = []
    for vec in test_vectors:
        # result is [(label, score), ...]
        pred = memory.query(vec, top_k=1)[0][0]
        predictions.append(pred)
        
    acc_oneshot = accuracy_score(y_test, predictions)
    print(f"One-Shot Accuracy: {acc_oneshot:.2%}")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # 4. Experiment B: Iterative Learning
    print("\n[Experiment B] Iterative Learning (Perceptron, 10 Epochs)")
    t0 = time.time()
    
    model = hv.PerceptronClassifier()
    model.fit(train_vectors, y_train, epochs=10, learning_rate=1)
    
    iter_preds = model.predict(test_vectors)
    acc_iter = accuracy_score(y_test, iter_preds)
    
    print(f"Iterative Accuracy: {acc_iter:.2%}")
    print(f"Time: {time.time() - t0:.2f}s")
    
    print("-" * 50)
    print(f"Improvement: +{acc_iter - acc_oneshot:.2%}")
    if acc_iter > 0.80:
        print(">> SUCCESS: Achieved >80% accuracy on complex image task!")
    else:
        print(">> NOTE: Accuracy is decent but could be higher with more data/epochs.")

if __name__ == "__main__":
    main()
