# 1. NLP (Natural Language Processing)

**NLP (Natural Language Processing)** to dziedzina sztucznej inteligencji zajmująca się przetwarzaniem i analizą języka naturalnego (mówionego i pisanego) przez komputery.  
Obejmuje m.in. takie zadania jak:
- Analiza sentymentu  
- Rozpoznawanie mowy  
- Tłumaczenie maszynowe  
- Tworzenie chatbotów  

---

# 2. Architektura Transformer

**Architektura Transformer** to rodzina modeli uczenia maszynowego, które przetwarzają sekwencje danych (np. słowa w zdaniu) równolegle, wykorzystując mechanizm *attention*.  
Kluczową rolę w Transformerach odgrywa **self-attention**, dzięki któremu model może rozumieć zależności między słowami niezależnie od ich położenia w sekwencji.  
Przykłady modeli opartych na Transformerach:
- BERT  
- GPT  
- XLM-R  

---

# 3. Model BERT, SQuAD, XLM-Roberta

- **BERT** (Bidirectional Encoder Representations from Transformers)  
  Model językowy oparty o architekturę Transformer, dwukierunkowy (*bidirectional*). Oznacza to, że rozumie kontekst słowa, patrząc zarówno w lewo, jak i w prawo w zdaniu.

- **SQuAD** (Stanford Question Answering Dataset)  
  Zbiór danych zawierający pytania i odpowiedzi na podstawie fragmentów tekstu. Jest standardem w trenowaniu i testowaniu modeli typu *question-answering*.

- **XLM-Roberta**  
  Wielojęzyczna wersja modelu Roberta, wytrenowana na wielu językach (w tym także na języku polskim). Może być wykorzystywana do różnych zadań NLP w trybie *multilingual*.

---

# 4. Embedding, przekształcanie na wektory

**Embedding** to reprezentacja słowa, zdania lub fragmentu tekstu w postaci wektora liczb w przestrzeni wielowymiarowej.  
Dzięki embeddingom komputer może „rozumieć” podobieństwa semantyczne między słowami – np. słowa „król” i „królowa” będą miały bliższe wektory niż „król” i „samolot”.

---

# 5. Podobieństwo kosinusowe

**Podobieństwo kosinusowe** to metryka używana do mierzenia podobieństwa między dwoma wektorami.  
Oblicza się je jako iloczyn skalarny podzielony przez iloczyn długości (norm) wektorów. Wynik mieści się w przedziale od -1 do 1, przy czym:
- 0 oznacza brak podobieństwa  
- 1 oznacza pełną zgodność (tożsame kierunki)  

W NLP najczęściej interesuje nas zakres od 0 do 1.

---

# 6. Model QA (Question Answering)

**Model QA (Question Answering)** to specjalny model NLP, który na podstawie kontekstu (np. fragmentu tekstu) i pytania generuje krótki fragment tekstu będący odpowiedzią.  
Modele tego typu znajdują zastosowanie m.in. w chatbotach i systemach wyszukiwania informacji.

---

# 7. Model transformacyjny

**Model transformacyjny** to ogólna nazwa dla modeli opartych na architekturze Transformer.  
Cechuje się zdolnością do przetwarzania sekwencji (np. zdań) efektywniej niż tradycyjne sieci rekurencyjne (RNN).

---

# 8. HAYSTACK

**HAYSTACK** to platforma (framework) do budowy systemów wyszukiwania i question-answering (tzw. *retriever + reader*).  
Umożliwia ona łączenie bazy danych/dokumentów z modelami QA, tak aby użytkownik mógł przeszukiwać duże zbiory tekstów i otrzymywać odpowiedzi w formie naturalnego języka.

---

# 9. Hugging Face

**Hugging Face** to serwis i społeczność zajmująca się udostępnianiem modeli NLP oraz narzędzi do uczenia maszynowego.  
W jego ramach dostępne są:
- Repozytorium z tysiącami modeli (*Hub*)  
- Biblioteka `transformers`  
- Platforma do trenowania i wdrażania modeli  

---

# 10. Fine Tuning (dostrajanie modelu)

**Fine Tuning** to proces dodatkowego trenowania wstępnie wytrenowanego modelu (np. BERT-a) na mniejszym, bardziej specjalistycznym zbiorze danych.  
Dzięki temu model może lepiej radzić sobie w konkretnym zadaniu, np. QA w języku polskim.

---

# 11. Pipeline (w kontekście transformers)

**Pipeline** (w kontekście biblioteki `transformers`) to wygodna abstrakcja (wysokopoziomowy interfejs) umożliwiająca szybkie użycie wytrenowanych modeli do konkretnych zadań, np. `pipeline("question-answering")`.  
Pozwala w kilku linijkach kodu wczytać model, tokenizer i wykonać predykcje (np. analizę sentymentu, QA czy tłumaczenie).

---
