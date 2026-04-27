"""
Script 1: Generatore di Dataset Sintetico
Hotel Review Classifier - Dataset Generator
Genera recensioni brevi sintetiche per hotel con etichette reparto e sentiment.
Output: reviews_dataset.csv con colonne: id, title, body, department, sentiment
"""

import csv
import random
import re

random.seed(42)

# ──────────────────────────────────────────────
# LESSICO PER REPARTO
# ──────────────────────────────────────────────

TEMPLATES = {
    "Housekeeping": {
        "pos": [
            ("Camera impeccabile", "La camera era pulitissima e il letto rifatto perfettamente. Asciugamani profumati e bagno brillante."),
            ("Pulizia eccellente", "Ho trovato la stanza in ottime condizioni: pavimento lucido, nessun capello, tutto in ordine."),
            ("Ottima igiene", "La pulizia della camera era al top: nessun odore, lenzuola fresche e cambiate ogni giorno."),
            ("Stanza fresca e ordinata", "Ogni mattina la camera veniva rifatta in modo impeccabile. Molto soddisfatto della pulizia."),
            ("Ottimo servizio pulizie", "Il personale delle pulizie è stato fantastico, sempre discreto e preciso nel lavoro."),
            ("Camera sempre in ordine", "Non ho mai trovato disordine. La camera era sempre pulita e profumata al mio ritorno."),
            ("Bagno perfetto", "Il bagno era pulitissimo, con dotazioni rinnovate ogni giorno. Massima cura nei dettagli."),
            ("Pulizia puntuale", "Le cameriere sono passate ogni mattina con puntualità. Camera sempre pronta e ordinata."),
        ],
        "neg": [
            ("Camera sporca", "La camera non era stata pulita a fondo: polvere sui mobili e pavimento non lavato."),
            ("Bagno trascurato", "Il bagno presentava aloni sul lavandino e capelli sulla doccia. Assolutamente inaccettabile."),
            ("Lenzuola non cambiate", "In tre giorni non hanno mai cambiato le lenzuola. Dovrò lamentarmi alla reception."),
            ("Cattivo odore in camera", "La camera aveva un odore sgradevole di umido e i cuscini sembravano vecchi."),
            ("Pulizia insufficiente", "Trovato residui di cibo sotto il letto e vetri del bagno sporchi. Deludente."),
            ("Asciugamani non sostituiti", "Non hanno mai cambiato gli asciugamani. Mi sono dovuto arrangiare per tutta la settimana."),
            ("Camera non rifatta", "Sono rientrato nel pomeriggio e la camera non era stata ancora pulita. Servizio lento."),
            ("Sporcizia nei dettagli", "La pulizia era superficiale: telecomando appiccicoso, bicchieri sul comodino rimasti da ieri."),
        ],
    },
    "Reception": {
        "pos": [
            ("Check-in velocissimo", "Il check-in è durato meno di cinque minuti. Staff cordiale e molto efficiente."),
            ("Personale disponibile", "Il personale della reception è stato sempre gentile e pronto a risolvere ogni problema."),
            ("Arrivo anticipato gestito bene", "Ho chiesto di fare il check-in prima dell'orario e me lo hanno permesso senza problemi."),
            ("Check-out rapido", "Il check-out è stato veloce e senza intoppi. Mi hanno anche preparato il conto in anticipo."),
            ("Accoglienza calorosa", "Al mio arrivo sono stato accolto con un sorriso e una spiegazione chiara dei servizi."),
            ("Ottimo supporto", "Ho avuto un problema con la chiave e lo staff lo ha risolto in pochi minuti. Ottimo."),
            ("Informazioni utili", "La receptionist mi ha fornito mappe, consigli sui ristoranti e prenotato un taxi. Perfetta."),
            ("Pagamento semplice", "Il pagamento è stato gestito in modo rapido e trasparente. Nessuna sorpresa sul conto."),
        ],
        "neg": [
            ("Attesa interminabile al check-in", "Ho aspettato più di 30 minuti per fare il check-in. Una sola persona allo sportello."),
            ("Staff scortese", "Il personale alla reception era freddo e scocciato. Non si sono nemmeno scusati per il ritardo."),
            ("Camera non pronta", "Arrivato alle 15 come da prenotazione ma la camera non era ancora pronta. Inaccettabile."),
            ("Check-out complicato", "Il check-out ha richiesto quasi un'ora a causa di errori nel conto. Esperienza stressante."),
            ("Problemi con prenotazione", "Hanno perso la mia prenotazione e mi hanno assegnato una camera diversa senza avvisarmi."),
            ("Nessuna risposta al telefono", "Ho chiamato la reception tre volte ma non hanno mai risposto. Pessimo servizio."),
            ("Personale non informato", "Ho chiesto informazioni sul bus e lo staff non sapeva nulla. Deludente e poco professionale."),
            ("Conto errato", "Al momento del check-out ho trovato addebiti non corretti. Ci è voluto molto tempo per sistemare."),
        ],
    },
    "F&B": {
        "pos": [
            ("Colazione deliziosa", "La colazione a buffet era ricchissima: frutta fresca, uova, dolci artigianali. Ottimo inizio giornata."),
            ("Ristorante eccellente", "La cena al ristorante dell'hotel era di ottima qualità. Piatti curati e porzioni generose."),
            ("Buffet abbondante", "Il buffet del mattino offriva tantissime scelte, tutte fresche. Soddisfatto ogni giorno."),
            ("Servizio bar impeccabile", "Il bar era sempre attivo e il barista preparava ottimi cocktail. Ambiente rilassante."),
            ("Cena romantica perfetta", "Abbiamo cenato nel ristorante interno: pesce freschissimo e un'ottima carta dei vini."),
            ("Colazione puntuale in camera", "Ho ordinato la colazione in camera ed è arrivata perfettamente in orario e ben presentata."),
            ("Menu vario", "Il menu offriva piatti tipici regionali e internazionali. Ho mangiato benissimo per tutta la settimana."),
            ("Staff del ristorante premuroso", "I camerieri erano attenti e gentili, suggerendo piatti e vini in modo professionale."),
        ],
        "neg": [
            ("Colazione deludente", "Il buffet era scarno: pane raffermo, succhi di frutta finti e nessuna scelta salata. Delusione."),
            ("Cibo freddo servito", "Il cibo al ristorante è arrivato freddo due sere di fila. Il personale non se n'è scusato."),
            ("Tempi di attesa lunghissimi", "Abbiamo aspettato quasi un'ora per essere serviti a cena. Inaccettabile per un hotel di questo livello."),
            ("Qualità scarsa", "I piatti sembravano pre-confezionati e riscaldati. Niente a che vedere con la descrizione nel menu."),
            ("Bar sempre chiuso", "Il bar era spesso chiuso anche nelle ore indicate. Ho dovuto uscire per trovare qualcosa da bere."),
            ("Colazione finita presto", "Arrivato alle 9 il buffet era quasi completamente esaurito. Rimasti solo gli avanzi."),
            ("Servizio lento", "I camerieri erano pochi e lentissimi. La cena si è trasformata in un'attesa estenuante."),
            ("Opzioni vegetariane assenti", "Nessun piatto vegetariano al ristorante. Per chi non mangia carne è stato un problema."),
        ],
    },
}

# Casi ambigui: tocca più reparti o ha sentiment misto
AMBIGUOUS = [
    ("Soggiorno altalenante", "Camera pulita ma check-in caotico. Il ristorante però ci ha salvato la serata con un'ottima cena.", "Housekeeping", "positive"),
    ("Hotel nel complesso ok", "La reception era lenta ma il personale gentile. Colazione nella norma, niente di speciale.", "Reception", "negative"),
    ("Esperienza mista", "Stanza sporca al primo giorno, poi sistemata. La colazione era buona ma il bar chiudeva presto.", "Housekeeping", "negative"),
    ("Staff ok, cibo no", "Il personale della reception ci ha accolto benissimo ma la cena al ristorante era pessima.", "F&B", "negative"),
    ("Ottima pulizia, tutto il resto meno", "La camera era impeccabile. Peccato per il check-in caotico e la colazione deludente.", "Housekeeping", "positive"),
    ("Servizi contrastanti", "Bagno pulitissimo, letto comodo. Il ristorante però lasciava a desiderare per qualità e tempi.", "F&B", "negative"),
    ("Un po' di tutto", "Il personale alla reception era scortese. La camera era però pulita e la colazione discreta.", "Reception", "negative"),
    ("Migliorabile", "Check-out veloce e staff gentile ma il ristorante aveva poco personale e il cibo era freddo.", "F&B", "negative"),
]

# ──────────────────────────────────────────────
# GENERAZIONE DATASET
# ──────────────────────────────────────────────

def generate_dataset(n_reviews=400, output_path="reviews_dataset.csv"):
    reviews = []
    review_id = 1

    # Proporzione: 40% Housekeeping, 30% Reception, 30% F&B
    dept_counts = {
        "Housekeeping": int(n_reviews * 0.40),
        "Reception": int(n_reviews * 0.30),
        "F&B": int(n_reviews * 0.30),
    }

    for dept, count in dept_counts.items():
        pos_list = TEMPLATES[dept]["pos"]
        neg_list = TEMPLATES[dept]["neg"]
        n_pos = count // 2
        n_neg = count - n_pos

        for i in range(n_pos):
            title, body = pos_list[i % len(pos_list)]
            # Leggera variazione nel testo
            body_var = body + (" Tornerò sicuramente." if i % 3 == 0 else "")
            reviews.append({
                "id": review_id,
                "title": title,
                "body": body_var,
                "department": dept,
                "sentiment": "positive"
            })
            review_id += 1

        for i in range(n_neg):
            title, body = neg_list[i % len(neg_list)]
            body_var = body + (" Non consiglio questo hotel." if i % 3 == 0 else "")
            reviews.append({
                "id": review_id,
                "title": title,
                "body": body_var,
                "department": dept,
                "sentiment": "negative"
            })
            review_id += 1

    # Aggiungi casi ambigui
    for title, body, dept, sent in AMBIGUOUS:
        reviews.append({
            "id": review_id,
            "title": title,
            "body": body,
            "department": dept,
            "sentiment": sent
        })
        review_id += 1

    random.shuffle(reviews)
    # Rinumera dopo shuffle
    for i, r in enumerate(reviews, 1):
        r["id"] = i

    # Salva CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "body", "department", "sentiment"])
        writer.writeheader()
        writer.writerows(reviews)

    print(f"✅ Dataset generato: {len(reviews)} recensioni → {output_path}")
    print(f"   Distribuzione reparto:")
    from collections import Counter
    dept_dist = Counter(r["department"] for r in reviews)
    sent_dist = Counter(r["sentiment"] for r in reviews)
    for k, v in dept_dist.items():
        print(f"     {k}: {v}")
    print(f"   Distribuzione sentiment:")
    for k, v in sent_dist.items():
        print(f"     {k}: {v}")
    return reviews


if __name__ == "__main__":
    generate_dataset(n_reviews=400, output_path="reviews_dataset.csv")
