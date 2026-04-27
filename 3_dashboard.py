"""
Script 3: Dashboard Interattiva — Hotel Review Classifier
Interfaccia Gradio per:
  - Analisi di una singola recensione (reparto + sentiment con probabilità)
  - Upload CSV batch → predizione → export con timestamp
Richiede: pip install gradio
"""

import pickle
import re
import os
import csv
import io
from datetime import datetime
import pandas as pd

# ──────────────────────────────────────────────
# PREPROCESSING (identico alla pipeline ML)
# ──────────────────────────────────────────────

STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","con","su",
    "per","tra","fra","e","o","ma","se","che","non","è","era","ho","ha","hanno",
    "al","del","nel","sul","dal","col","mi","ti","si","ci","vi","li","le","ne",
    "come","anche","più","molto","poco","già","mai","sempre","però","quindi",
    "ancora","questo","questa","questi","queste","tutto","tutti","tutta","tutte",
    "ogni","nessun","nessuna","qualche","mio","mia","miei","suo","sua","suoi",
    "loro","nostro","vostro","essere","avere","fare","sono","stato","stata"
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_IT and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# CARICAMENTO MODELLI
# ──────────────────────────────────────────────

def load_models():
    if not os.path.exists("dept_model.pkl") or not os.path.exists("sent_model.pkl"):
        raise FileNotFoundError(
            "Modelli non trovati. Esegui prima:\n"
            "  python 1_generate_dataset.py\n"
            "  python 2_ml_pipeline.py"
        )
    with open("dept_model.pkl", "rb") as f:
        dept_model = pickle.load(f)
    with open("sent_model.pkl", "rb") as f:
        sent_model = pickle.load(f)
    return dept_model, sent_model


dept_model, sent_model = load_models()

DEPT_EMOJI = {"Housekeeping": "🧹", "Reception": "🛎️", "F&B": "🍽️"}
SENT_EMOJI = {"positive": "😊", "negative": "😞"}


# ──────────────────────────────────────────────
# FUNZIONE: ANALISI SINGOLA RECENSIONE
# ──────────────────────────────────────────────

def analyze_review(title: str, body: str):
    if not title.strip() and not body.strip():
        return "⚠️ Inserisci almeno un titolo o un testo.", ""

    full_text = f"{title} {body}".strip()
    clean = preprocess_text(full_text)

    # Predizione reparto
    dept_pred = dept_model.predict([clean])[0]
    dept_proba = dept_model.predict_proba([clean])[0]
    dept_classes = dept_model.classes_

    # Predizione sentiment
    sent_pred = sent_model.predict([clean])[0]
    sent_proba = sent_model.predict_proba([clean])[0]
    sent_classes = sent_model.classes_

    dept_conf = dept_proba.max() * 100
    sent_conf = sent_proba.max() * 100

    # Output principale
    result = (
        f"## Risultato Analisi\n\n"
        f"**Reparto consigliato:** {DEPT_EMOJI.get(dept_pred, '')} **{dept_pred}**  "
        f"(confidenza: `{dept_conf:.1f}%`)\n\n"
        f"**Sentiment:** {SENT_EMOJI.get(sent_pred, '')} **{sent_pred.capitalize()}**  "
        f"(confidenza: `{sent_conf:.1f}%`)\n\n"
        f"---\n"
        f"**Probabilità per reparto:**\n"
    )
    for cls, prob in sorted(zip(dept_classes, dept_proba), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        result += f"- {DEPT_EMOJI.get(cls, '')} {cls}: `{prob*100:.1f}%` {bar}\n"

    result += f"\n**Probabilità sentiment:**\n"
    for cls, prob in sorted(zip(sent_classes, sent_proba), key=lambda x: -x[1]):
        result += f"- {SENT_EMOJI.get(cls, '')} {cls.capitalize()}: `{prob*100:.1f}%`\n"

    # Raccomandazione azione
    action = ""
    if sent_pred == "negative":
        if dept_pred == "Housekeeping":
            action = "🔔 **Azione suggerita:** Notificare responsabile Housekeeping entro 2h. Offrire cambio camera o scuse formali."
        elif dept_pred == "Reception":
            action = "🔔 **Azione suggerita:** Responsabile Reception contatta il cliente entro 1h. Verificare il processo di check-in/out."
        elif dept_pred == "F&B":
            action = "🔔 **Azione suggerita:** Notificare F&B Manager. Valutare compensazione (drink omaggio, colazione gratuita)."
    else:
        action = "✅ **Azione suggerita:** Nessuna urgenza. Considerare risposta pubblica di ringraziamento."

    return result, action


# ──────────────────────────────────────────────
# FUNZIONE: PREDIZIONE BATCH DA CSV
# ──────────────────────────────────────────────

def predict_batch(file):
    if file is None:
        return None, "⚠️ Carica un file CSV."

    try:
        df = pd.read_csv(file.name, encoding="utf-8")
    except Exception as e:
        return None, f"❌ Errore lettura file: {e}"

    required = {"title", "body"}
    if not required.issubset(set(df.columns)):
        return None, f"❌ Il CSV deve avere almeno le colonne: `title`, `body`. Trovate: {list(df.columns)}"

    df["text_clean"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(preprocess_text)

    df["pred_department"] = dept_model.predict(df["text_clean"])
    df["pred_sentiment"] = sent_model.predict(df["text_clean"])

    dept_proba = dept_model.predict_proba(df["text_clean"])
    sent_proba = sent_model.predict_proba(df["text_clean"])
    df["dept_confidence"] = dept_proba.max(axis=1).round(3)
    df["sent_confidence"] = sent_proba.max(axis=1).round(3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"batch_predictions_{timestamp}.csv"

    cols = ["title", "body"]
    if "id" in df.columns:
        cols = ["id"] + cols
    if "department" in df.columns:
        cols += ["department"]
    if "sentiment" in df.columns:
        cols += ["sentiment"]
    cols += ["pred_department", "pred_sentiment", "dept_confidence", "sent_confidence"]

    df[cols].to_csv(output_path, index=False)

    summary = (
        f"✅ **Predizioni completate!**\n\n"
        f"- Recensioni processate: `{len(df)}`\n"
        f"- Timestamp: `{timestamp}`\n"
        f"- File salvato: `{output_path}`\n\n"
        f"**Distribuzione reparto predetto:**\n"
    )
    for dept, count in df["pred_department"].value_counts().items():
        summary += f"- {DEPT_EMOJI.get(dept,'')} {dept}: {count} ({count/len(df)*100:.0f}%)\n"
    summary += "\n**Distribuzione sentiment predetto:**\n"
    for sent, count in df["pred_sentiment"].value_counts().items():
        summary += f"- {SENT_EMOJI.get(sent,'')} {sent.capitalize()}: {count} ({count/len(df)*100:.0f}%)\n"

    return output_path, summary


# ──────────────────────────────────────────────
# INTERFACCIA GRADIO
# ──────────────────────────────────────────────

def build_interface():
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio non installato. Esegui: pip install gradio --break-system-packages")
        return

    with gr.Blocks(title="Hotel Review Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🏨 Hotel Review Classifier\n"
            "Classifica automaticamente le recensioni per **reparto** (Housekeeping / Reception / F&B) "
            "e **sentiment** (positivo / negativo) usando Machine Learning."
        )

        with gr.Tab("📝 Analisi Singola Recensione"):
            with gr.Row():
                with gr.Column():
                    title_input = gr.Textbox(
                        label="Titolo recensione",
                        placeholder="Es: Camera sporca, Check-in veloce..."
                    )
                    body_input = gr.Textbox(
                        label="Testo recensione",
                        placeholder="Es: La camera non era stata pulita...",
                        lines=5
                    )
                    analyze_btn = gr.Button("🔍 Analizza", variant="primary")

                with gr.Column():
                    result_output = gr.Markdown(label="Risultato")
                    action_output = gr.Markdown(label="Azione suggerita")

            analyze_btn.click(
                fn=analyze_review,
                inputs=[title_input, body_input],
                outputs=[result_output, action_output]
            )

            gr.Examples(
                examples=[
                    ["Camera sporca", "La camera non era stata pulita a fondo. Trovato polvere e capelli nel bagno."],
                    ["Check-in velocissimo", "Il check-in è durato meno di cinque minuti. Staff molto cordiale."],
                    ["Colazione ottima", "Il buffet era ricchissimo con frutta fresca e dolci artigianali."],
                    ["Esperienza mista", "Camera pulita ma il check-in è stato caotico. La colazione però ottima."],
                ],
                inputs=[title_input, body_input],
                label="Esempi rapidi"
            )

        with gr.Tab("📂 Predizione Batch (CSV)"):
            gr.Markdown(
                "Carica un file CSV con le colonne `title` e `body`. "
                "Il file risultante includerà reparto e sentiment predetti con confidenza."
            )
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Carica CSV", file_types=[".csv"])
                    batch_btn = gr.Button("⚙️ Processa Batch", variant="primary")
                with gr.Column():
                    batch_summary = gr.Markdown(label="Riepilogo")
                    file_output = gr.File(label="Download CSV risultati")

            batch_btn.click(
                fn=predict_batch,
                inputs=[file_input],
                outputs=[file_output, batch_summary]
            )

        gr.Markdown(
            "---\n"
            "*Hotel Review Classifier — Pipeline ML con Logistic Regression + TF-IDF*  \n"
            "*Sviluppato con Python, scikit-learn, Gradio*"
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    if demo:
        print("🚀 Avvio dashboard su http://localhost:7860")
        demo.launch(share=False)
