# evaluation.py (Versione DEFINITIVA a prova di errore)

import os
import sys
import json
import time
# --- INIZIO SOLUZIONE DEFINITIVA PER L'IMPORT ---
# Ottiene il percorso assoluto della cartella 'code' in cui si trova questo script
# e lo aggiunge ai percorsi in cui Python cerca i moduli.
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)
# --- FINE SOLUZIONE ---

# Ora questi import DEVONO funzionare
from index import product
from writer import writer_gemini
# ---

import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("--- INIZIO FASE C: VALUTAZIONE DEL MODELLO RAG ---")

# Carica le variabili d'ambiente
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- 1. CARICAMENTO DEL GOLDEN DATASET ---
print("📚 Caricamento del golden_dataset.json...")
try:
    # Il percorso deve essere relativo alla cartella principale del progetto
    dataset_path = os.path.join(script_directory, 'golden_dataset.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        golden_dataset = json.load(f)
    print(f"✅ Dataset caricato con successo! {len(golden_dataset)} esempi trovati.")
except FileNotFoundError:
    print(f"❌ ERRORE: File '{dataset_path}' non trovato. Assicurati che 'golden_dataset.json' sia nella cartella 'code'.")
    sys.exit() # Interrompe l'esecuzione se il dataset non è trovato


# --- 2. CARICAMENTO DEL MODELLO PER LA SIMILARITÀ SEMANTICA ---
print("🔎 Caricamento del modello di embedding per la valutazione...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello caricato.")


# --- 3. FUNZIONE DI ORCHESTRAZIONE RAG ---
def esegui_rag(domanda: str) -> str:
    print(f"   - Eseguo il RAG per la domanda: '{domanda[:40]}...'")
    documenti_trovati = product.find_products(context=domanda, top=3)
    if not documenti_trovati:
        contesto_per_prompt = "Nessun contesto specifico trovato."
    else:
        contesto_per_prompt = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documenti_trovati]
        )
    risposta_generata = writer_gemini.write(
        productContext=contesto_per_prompt,
        assignment=domanda,
    )
    return risposta_generata


# --- 4. CICLO DI VALUTAZIONE ---
risultati_valutazione = []

print("\n🚀 Avvio del ciclo di valutazione sul Golden Dataset...")
for i, item in enumerate(golden_dataset):
    print(f"\n--- Valutando l'item {i+1}/{len(golden_dataset)} (Categoria: {item['categoria']}) ---")
    domanda_test = item["domanda"]
    risposta_ideale = item["risposta_ideale"]

    risposta_generata = esegui_rag(domanda_test)
    
    embedding_generato = model.encode([risposta_generata])
    embedding_ideale = model.encode([risposta_ideale])

    score = cosine_similarity(embedding_generato, embedding_ideale)[0][0]
    risultati_valutazione.append(score)
    
    print(f"   - Punteggio di Similarità Semantica: {score:.4f}")
    time.sleep(2)


# --- 5. RISULTATO FINALE ---
punteggio_medio = sum(risultati_valutazione) / len(risultati_valutazione)

print("\n\n--- RISULTATI FINALI DELLA VALUTAZIONE ---")
print(f"Numero di campioni di test: {len(golden_dataset)}")
print(f"Punteggi individuali: {[f'{s:.4f}' for s in risultati_valutazione]}")
print(f"📈 Punteggio Medio di Similarità Semantica del Modello: {punteggio_medio:.4f}")