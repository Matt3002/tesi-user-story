# crea_indice_api.py (Versione con Chiamata API Diretta)

import os
import json
import requests # Usiamo la libreria standard per le chiamate HTTP
from dotenv import load_dotenv

print("✅ ESEGUO SCRIPT DI CREAZIONE INDICE TRAMITE API DIRETTA.")
print("--------------------------------------------------")

load_dotenv()

# --- 1. CONFIGURAZIONE ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
API_VERSION = "2023-11-01" # Versione stabile dell'API di Azure Search

if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME]):
    raise ValueError("Variabili d'ambiente non trovate. Controlla il file .env")

# --- 2. PREPARAZIONE DELLA RICHIESTA API ---

# L'URL completo per la creazione/aggiornamento di un indice
url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version={API_VERSION}"

# Le intestazioni (headers) richieste per l'autenticazione
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_API_KEY
}

# Il corpo (body) della richiesta, che è il nostro dizionario/JSON
index_body = {
    "name": AZURE_SEARCH_INDEX_NAME,
    "fields": [
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        {"name": "content", "type": "Edm.String", "searchable": True},
        {
            "name": "content_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": 768,
            "vectorSearchProfile": "my-hnsw-profile",
        },
    ],
    "vectorSearch": {
        "profiles": [
            {"name": "my-hnsw-profile", "algorithm": "my-hnsw-config"}
        ],
        "algorithms": [
            {"name": "my-hnsw-config", "kind": "hnsw"}
        ],
    },
}

# --- 3. ESECUZIONE DELLA CHIAMATA API ---
print(f"📡 Invio richiesta PUT all'URL: {url}")

# Usiamo una richiesta PUT, che è l'operazione standard per creare o sostituire una risorsa
response = requests.put(url, headers=headers, data=json.dumps(index_body))

# --- 4. CONTROLLO DEL RISULTATO ---
if response.status_code == 200 or response.status_code == 201:
    # 200 OK (se l'indice è stato aggiornato) o 201 Created (se è stato creato)
    print("\n🎉🎉🎉 CONGRATULAZIONI! L'INDICE È STATO CREATO CON SUCCESSO! 🎉🎉🎉")
    print("Lo scontro con la libreria è finalmente vinto.")
else:
    # Se c'è un errore, Azure ci risponderà con un JSON che lo descrive
    print(f"\n❌ ERRORE! Azure ha risposto con codice: {response.status_code}")
    print("Dettagli dell'errore dal server di Azure:")
    print(response.json())