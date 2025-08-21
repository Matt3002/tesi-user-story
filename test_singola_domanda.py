# run_gemini_test.py (o come preferisci chiamarlo)

import os
import json
from pathlib import Path
# Assumiamo di avere una funzione per recuperare i documenti, come nel tuo progetto
from index import product # La tua funzione che interroga Azure AI Search
from writers import writer_gemini # Il nostro nuovo writer per Gemini
from dotenv import load_dotenv

# Carica le variabili d'ambiente (GEMINI_API_KEY, AZURE_SEARCH_API_KEY, etc.)
load_dotenv()

# --- ORCHESTRAZIONE DEL FLUSSO ---

# 1. Domanda dell'utente
domanda_utente = "Voglio una user story per il login con autenticazione a due fattori"

# 2. Recupera il contesto pertinente da Azure AI Search
print("🔎 Recupero il contesto da Azure AI Search...")
# La funzione find_products esegue la ricerca vettoriale e restituisce i documenti
# (La logica di embedding della domanda è dentro find_products)
documenti_trovati = product.find_products(context=domanda_utente, top=3)

# Prepara il contesto per il prompt
if not documenti_trovati:
    print("⚠️ Nessun documento trovato.")
    contesto_per_prompt = "Nessun contesto specifico trovato nella base di conoscenza."
else:
    # Formattiamo il contesto come una singola stringa di testo
    contesto_per_prompt = "\n\n---\n\n".join(
        [doc.get("content", "") for doc in documenti_trovati]
    )

# 3. Chiama il writer di Gemini per generare la risposta
print("🧠 Genero la risposta con Gemini tramite Prompty...")
# Passiamo il contesto e la domanda al nostro writer
risposta_finale = writer_gemini.write(
    productContext=contesto_per_prompt,
    assignment=domanda_utente
)

# 4. Stampa il risultato
print("\n✨ USER STORY GENERATA DA GEMINI (via Prompty) ✨")
print(risposta_finale)