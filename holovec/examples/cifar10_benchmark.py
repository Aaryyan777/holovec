import holovec as hv
import numpy as np
import urllib.request
import tarfile
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def download_and_load_cifar10():
    """
    Manually downloads and extracts CIFAR-10 dataset.
    Returns X (images) and y (labels).
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    if not os.path.exists(filename):
        print(f"Downloading CIFAR-10 from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
        
    print("Extracting CIFAR-10...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
        
    data_dir = "cifar-10-batches-py"
    
    # Load batch 1 (10,000 images) - sufficient for this benchmark
    # We could load all 5 batches, but that's 50k images -> heavy for CPU-only HDC demo
    with open(os.path.join(data_dir, "data_batch_1"), 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data'] # (10000, 3072)
        y = np.array(dict[b'labels'])
        
    return X, y

def main():
    print("=== HoloVec: CIFAR-10 Benchmark (The 'Step Up' Challenge) ===")
    print("Task: Classify 32x32 Color Images (Airplane, Bird, Cat, etc.)")
    print("Input Dimensions: 3,072 (High-Dim Color Data)")
    print("Strategy: ProjectionEncoder -> Iterative Perceptron")
    print("-" * 50)

    # 1. Load Data
    try:
        X, y = download_and_load_cifar10()
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return

    # Normalize (0-1)
    X = X / 255.0
    
    # Use a subset of 2,000 training images to keep the demo quick (< 1 min)
    # CIFAR-10 is heavy. Projection of 3072 -> 10,000 dims is 30M ops per image.
    TRAIN_SIZE = 2000
    TEST_SIZE = 500
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    
    print(f"\nData Loaded. Training on {TRAIN_SIZE} images, Testing on {TEST_SIZE}.")
    print("Classes: 0:Airplane, 1:Auto, 2:Bird, 3:Cat, 4:Deer, 5:Dog, 6:Frog, 7:Horse, 8:Ship, 9:Truck")

    # 2. Encode
    print("\n[Step 1] Projecting 3,072 features to 10,000-D HyperVectors...")
    print("Note: This involves a large matrix multiplication (3072x10000). Please wait.")
    t0 = time.time()
    
    encoder = hv.ProjectionEncoder(input_dim=3072)
    
    print("  Encoding Training Set...")
    train_vectors = encoder.encode_batch(X_train)
    
    print("  Encoding Test Set...")
    test_vectors = encoder.encode_batch(X_test)
    
    print(f"Encoding finished in {time.time() - t0:.2f} seconds.")

    # 3. Experiment A: One-Shot
    print("\n[Experiment A] One-Shot Learning")
    t0 = time.time()
    memory = hv.AssociativeMemory()
    
    class_data = {}
    for vec, label in zip(train_vectors, y_train):
        if label not in class_data: class_data[label] = []
        class_data[label].append(vec)
        
    for label, vecs in class_data.items():
        memory.add(label, hv.HyperVector.bundle_all(vecs))
        
    preds_oneshot = [memory.query(v, top_k=1)[0][0] for v in test_vectors]
    acc_oneshot = accuracy_score(y_test, preds_oneshot)
    
    print(f"One-Shot Accuracy: {acc_oneshot:.2%}")
    print(f"Time: {time.time() - t0:.2f}s")

    # 4. Experiment B: Iterative (The Real Test)
    EPOCHS = 15
    print(f"\n[Experiment B] Iterative Learning ({EPOCHS} Epochs)")
    t0 = time.time()
    
    model = hv.PerceptronClassifier()
    model.fit(train_vectors, y_train, epochs=EPOCHS, learning_rate=1)
    
    preds_iter = model.predict(test_vectors)
    acc_iter = accuracy_score(y_test, preds_iter)
    
    print(f"Iterative Accuracy: {acc_iter:.2%}")
    print(f"Time: {time.time() - t0:.2f}s")
    
    print("-" * 50)
    print(f"Improvement: +{acc_iter - acc_oneshot:.2%}")
    print(f"Baseline Random Guessing: 10.00%")
    if acc_iter > 0.35:
        print(">>> SUCCESS: Beating standard linear classifiers (~30-35%) on raw pixels!")

if __name__ == "__main__":
    main()
