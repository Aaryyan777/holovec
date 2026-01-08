import sys
import os

# Add parent directory to path to import local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoders import TextEncoder
from src.memory import AssociativeMemory
from src.core import HyperVector

def main():
    print("=== HoloVec: Hyperdimensional Computing Demo ===")
    print("Task: Language Identification (One-Shot Learning)")
    print("-" * 50)

    # 1. Initialize
    encoder = TextEncoder(ngram_size=3)
    brain = AssociativeMemory()

    # 2. Training Data (Just a few sentences!)
    training_data = {
        "English": [
            "The quick brown fox jumps over the lazy dog",
            "I love machine learning libraries",
            "Hello world this is a test"
        ],
        "Spanish": [
            "El veloz zorro marrón salta sobre el perro perezoso",
            "Me encantan las bibliotecas de aprendizaje automático",
            "Hola mundo esto es una prueba"
        ],
        "German": [
            "Der schnelle braune Fuchs springt über den faulen Hund",
            "Ich liebe Bibliotheken für maschinelles Lernen",
            "Hallo Welt, das ist ein Test"
        ],
        "French": [
            "Le renard brun rapide saute par-dessus le chien paresseux",
            "J'adore les bibliothèques d'apprentissage automatique",
            "Bonjour le monde c'est un test"
        ]
    }

    # 3. "Training" Phase
    print("Training...", end=" ")
    for lang, sentences in training_data.items():
        # Create a single prototype vector for the language
        # by bundling all sentence vectors together with EQUAL weight
        sentence_vectors = [encoder.encode(s) for s in sentences]
        lang_vector = HyperVector.bundle_all(sentence_vectors)
        
        brain.add(lang, lang_vector)
    print("Done! (Instantaneous)")

    # 4. Testing Phase
    test_sentences = [
        "Machine learning is fascinating", # English
        "Esto es muy interesante",         # Spanish
        "Das ist sehr interessant",        # German
        "C'est très intéressant",          # French
        "The dog runs fast"                # English
    ]

    print("-" * 50)
    print("Testing on new unseen sentences:")
    print("-" * 50)

    for sent in test_sentences:
        # Encode query
        query_vec = encoder.encode(sent)
        
        # Query memory
        prediction = brain.query(query_vec, top_k=1)[0]
        label, confidence = prediction
        
        print(f"Input: '{sent}'")
        print(f"Predicted: {label} (Confidence: {confidence:.4f})")
        print()

if __name__ == "__main__":
    main()
