# 🏨 Hotel Review Classifier

Prototipo ML per la classificazione automatica di recensioni hotel per **reparto** e **sentiment**
(sviluppato con Python 3, scikit-learn, Gradio).
---

## 📁 Struttura del progetto

```
hotel_review_classifier/
├── 1_generate_dataset.py     # Genera il dataset sintetico (CSV)
├── 2_ml_pipeline.py          # Pipeline ML: preprocessing, training, valutazione, grafici
├── 3_dashboard.py            # Dashboard interattiva Gradio
├── reviews_dataset.csv       # Dataset sintetico (generato automaticamente)
├── predictions_batch.csv     # Output predizioni su tutto il dataset
├── dept_model.pkl            # Modello classificatore reparto (salvato)
├── sent_model.pkl            # Modello classificatore sentiment (salvato)
├── confusion_matrix_dept.png # Grafico confusion matrix reparto
├── confusion_matrix_sent.png # Grafico confusion matrix sentiment
├── f1_bar_chart.png          # Grafico F1-score per classe
├── metrics.json              # Metriche finali in formato JSON
└── README.md                 # Questo file
```

---

## ⚙️ Requisiti

- Python 3.9+
- pip packages:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
pip install gradio          # solo per il dashboard (Script 3)
```

---

## 🚀 Come eseguire (in ordine)

### Step 1 — Genera il dataset sintetico

```bash
python 1_generate_dataset.py
```

**Output:** `reviews_dataset.csv` con 408 recensioni e colonne:
`id`, `title`, `body`, `department`, `sentiment`

---

### Step 2 — Addestra i modelli e valuta

```bash
python 2_ml_pipeline.py
```

**Output:**
- Stampa accuracy, F1 macro, classification report per reparto e sentiment
- `confusion_matrix_dept.png` — confusion matrix per reparto
- `confusion_matrix_sent.png` — confusion matrix per sentiment
- `f1_bar_chart.png` — F1-score per classe (bar chart)
- `predictions_batch.csv` — predizioni su tutto il dataset con confidenza
- `dept_model.pkl`, `sent_model.pkl` — modelli serializzati
- `metrics.json` — metriche riassuntive

---

### Step 3 — Dashboard interattiva

```bash
python 3_dashboard.py
```

Apri il browser su: **http://localhost:7860**

**Funzionalità:**
- **Tab 1:** Analisi singola recensione → reparto + sentiment con probabilità + azione suggerita
- **Tab 2:** Upload CSV batch → predizioni → download risultati con timestamp

---

## 📊 Risultati ottenuti

| Modello         | Accuracy | F1 Macro |
|-----------------|----------|----------|
| Reparto (3 cl.) | 98.8%    | 98.8%    |
| Sentiment (bin.)| 100.0%   | 100.0%   |

> **Nota:** Le alte performance dipendono dal dataset sintetico con lessico controllato.
> Su dati reali (OTA, email) si attende una riduzione del 10-20% in entrambe le metriche.

---

## 🔧 Parametri principali

| Parametro        | Valore        | File              |
|------------------|---------------|-------------------|
| n_reviews        | 400 (+8 amb.) | 1_generate_dataset.py |
| train/test split | 80/20         | 2_ml_pipeline.py  |
| TF-IDF ngram     | (1,2)         | 2_ml_pipeline.py  |
| TF-IDF features  | 5000          | 2_ml_pipeline.py  |
| LR C (regol.)    | 1.0           | 2_ml_pipeline.py  |
| LR max_iter      | 500           | 2_ml_pipeline.py  |

---

## ⚠️ Limiti noti

1. **Dataset sintetico** — lessico ripetitivo, non cattura variazioni reali
2. **Stopword italiano** — lista manuale, non completa; usare NLTK per casi reali
3. **No lemmatizzazione** — "pulita/pulizia/pulire" trattati come token diversi
4. **Casi ambigui** — recensioni multi-reparto assegnate a un solo reparto
5. **Soglie di confidenza** — nessuna gestione di predizioni a bassa confidenza

---

## 📝 Formato CSV per predizioni batch

Il CSV di input deve avere almeno le colonne:

```
title,body
"Camera sporca","Il bagno non era pulito..."
"Check-in rapido","Ho trovato tutto pronto..."
```

Colonne opzionali: `id`, `department`, `sentiment` (usate per confronto se presenti)

---

## 👤 Autore

1. **Nome e Cognome:** Francesco Bologna
2. **Matricola:** 0312300081
3. **Corso di Laurea:** Informatica per le Aziende Digitali (L-31)
