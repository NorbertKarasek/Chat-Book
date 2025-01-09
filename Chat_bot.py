import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os

# Wyłącz ostrzeżenia o dowiązaniach symbolicznych dla Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Funkcja do dzielenia tekstu na fragmenty o określonym rozmiarze
def podziel_tekst_na_chunki(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

# Funkcja do obliczania podobieństwa kosinusowego między dwoma wektorami
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # Ścieżka do pliku tekstowego
    sciezka_do_pliku = "Krolowa_sniegu.txt"

    # Odczyt pliku tekstowego
    with open(sciezka_do_pliku, "r", encoding="utf-8") as f:
        text = f.read()

    # Podział tekstu na fragmenty
    chunks = podziel_tekst_na_chunki(text, chunk_size=500)
    print(f"Liczba fragmentów (chunków): {len(chunks)}")

    # Ładowanie modelu do embeddingów
    print("Ładowanie modelu do embeddingów...")
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Ładowanie modelu QA dla języka polskiego
    print("Ładowanie modelu QA (język polski)...")
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/xlm-roberta-base-squad2",
        tokenizer="deepset/xlm-roberta-base-squad2"
    )

    # Obliczanie embeddingów dla fragmentów tekstu
    print("Obliczanie embeddingów dla fragmentów tekstu...")
    chunk_embeddings = embed_model.encode(chunks)

    print("\nChatbot gotowy! Możesz zadawać pytania dotyczące treści książki.")
    print("Wpisz 'koniec' aby zakończyć.\n")

    while True:
        # Pobranie pytania od użytkownika
        user_question = input("Twoje pytanie: ")

        # Wyjście z pętli, jeśli użytkownik wpisze 'koniec'
        if user_question.lower() == "koniec":
            print("Zakończono działanie chatbota.")
            break

        # Obliczanie embeddingu dla pytania użytkownika
        question_embedding = embed_model.encode([user_question])[0]

        # Obliczanie podobieństw między embeddingiem pytania a embeddingami fragmentów
        similarities = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]

        # Znalezienie fragmentu z najwyższym podobieństwem
        best_chunk_idx = np.argmax(similarities)
        best_chunk = chunks[best_chunk_idx]

        # Uzyskanie odpowiedzi z modelu QA
        answer = qa_pipeline(question=user_question, context=best_chunk)

        # Wyświetlenie odpowiedzi i punktacji modelu
        print("\nOdpowiedź:", answer['answer'])
        print("Punktacja modelu (score):", answer['score'])
        print("-" * 70)

if __name__ == "__main__":
    main()