import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

def podziel_tekst_na_chunki(text, chunk_size=500):
    """
    Dzieli tekst na fragmenty ('chunki') o zadanej liczbie słów.
    Zwraca listę stringów.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def cosine_similarity(a, b):
    """
    Zwraca podobieństwo kosinusowe między dwoma wektorami.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # 1. Wczytanie treści książki z pliku .txt
    sciezka_do_pliku = "ksiazka.txt"  # <-- zmień tę ścieżkę na swoją
    with open(sciezka_do_pliku, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Podział tekstu na chunki (domyślnie po 500 słów)
    chunks = podziel_tekst_na_chunki(text, chunk_size=500)
    print(f"Liczba fragmentów (chunków): {len(chunks)}")

    # 3. Ładowanie modelu do embeddingów (wielojęzycznego) i modelu QA w języku polskim
    print("Ładowanie modelu do embeddingów...")
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Ładowanie modelu QA (język polski)...")
    qa_pipeline = pipeline(
        "question-answering",
        model="dkleczek/bert-base-polish-cased-v1-squad2",
        tokenizer="dkleczek/bert-base-polish-cased-v1-squad2"
    )

    # 4. Oblicz embeddingi dla wszystkich chunków
    print("Obliczanie embeddingów dla fragmentów tekstu...")
    chunk_embeddings = embed_model.encode(chunks)

    # 5. Pętla główna - zadawanie pytań użytkownika
    print("\nChatbot gotowy! Możesz zadawać pytania dotyczące treści książki.")
    print("Wpisz 'koniec' aby zakończyć.\n")

    while True:
        user_question = input("Twoje pytanie: ")

        if user_question.lower() == "koniec":
            print("Zakończono działanie chatbota.")
            break

        # 5a. Obliczamy embedding pytania
        question_embedding = embed_model.encode([user_question])[0]

        # 5b. Znajdujemy chunk z najwyższą wartością podobieństwa
        similarities = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]
        best_chunk_idx = np.argmax(similarities)
        best_chunk = chunks[best_chunk_idx]

        # 5c. Używamy modelu QA, aby wydobyć odpowiedź z wybranego fragmentu
        answer = qa_pipeline({
            "question": user_question,
            "context": best_chunk
        })

        # 6. Wyświetlamy wynik
        print("\nOdpowiedź:", answer['answer'])
        print("Punktacja modelu (score):", answer['score'])
        print("-" * 70)

if __name__ == "__main__":
    main()
