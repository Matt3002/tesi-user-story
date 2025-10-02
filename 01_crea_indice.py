# 01_crea_indice.py
# This script is responsible for creating or updating the index schema in Azure AI Search.
# It defines the structure of the data that will be stored, including the fields for content,
# ID, and the crucial vector field for semantic search. This script should typically be
# run only once at the beginning of the project setup.

import os
import json
import requests
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env.
load_dotenv()

# --- Configurazione di Azure AI Search ---
# Recupera le credenziali e gli identificatori necessari dalle variabili d'ambiente.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
# Usa una versione stabile dell'API per coerenza e per evitare modifiche che potrebbero rompere il codice.
API_VERSION = "2023-11-01"

# Valida che tutte le variabili d'ambiente richieste siano impostate prima di procedere.
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME]):
    raise ValueError("Una o più variabili d'ambiente non sono impostate. Controlla il tuo file .env.")

# Costruisce l'URL completo per l'endpoint dell'API REST per creare/aggiornare un indice.
url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version={API_VERSION}"

# Imposta gli header richiesti per la richiesta API, inclusi il tipo di contenuto e la chiave API per l'autenticazione.
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_API_KEY
}

# --- Definizione dello Schema dell'Indice ---
# Questo oggetto JSON definisce la struttura (schema) dell'indice di ricerca. Specifica i campi,
# i loro tipi e la configurazione della ricerca vettoriale.
index_body = {
    "name": AZURE_SEARCH_INDEX_NAME,
    "fields": [
        # Il campo 'id' è l'identificatore univoco per ogni documento (chunk).
        # 'key': True lo rende la chiave primaria.
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        
        # Il campo 'content' contiene il testo effettivo del chunk.
        # 'searchable': True permette la ricerca full-text su questo campo.
        {"name": "content", "type": "Edm.String", "searchable": True},
        
        # Il campo 'content_vector' memorizza l'embedding vettoriale del contenuto.
        # 'type': 'Collection(Edm.Single)' specifica un array di numeri float a precisione singola.
        {
            "name": "content_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            # 'dimensions' DEVE corrispondere alle dimensioni di output del modello di embedding.
            # 'paraphrase-multilingual-MiniLM-L12-v2' produce vettori a 384 dimensioni.
            "dimensions": 384,
            "vectorSearchProfile": "my-hnsw-profile",
        },
    ],
    "vectorSearch": {
        # 'profiles' definisce configurazioni nominate per la ricerca vettoriale.
        "profiles": [
            {"name": "my-hnsw-profile", "algorithm": "my-hnsw-config"}
        ],
        # 'algorithms' definisce gli algoritmi di ricerca vettoriale. HNSW (Hierarchical Navigable Small World)
        # è un algoritmo ad alte prestazioni utilizzato per la ricerca approssimata dei vicini più prossimi.
        "algorithms": [
            {"name": "my-hnsw-config", "kind": "hnsw"}
        ],
    },
}

print(f"Invio della richiesta PUT per creare/aggiornare l'indice a: {url}")

# Invia la richiesta ad Azure AI Search per creare o aggiornare l'indice.
# Una richiesta PUT è idempotente, il che significa che può essere eseguita più volte in sicurezza. Se l'indice
# esiste già con questo schema, Azure confermerà il successo. Se esiste ma con uno
# schema diverso, sarà aggiornato. Se non esiste, sarà creato.
response = requests.put(url, headers=headers, data=json.dumps(index_body))

# Controlla il codice di stato della risposta HTTP per confermare il risultato.
# I codici di stato 200 (OK) o 201 (Created) indicano il successo.
if response.status_code in [200, 201]:
    print("\nSUCCESSO: L'indice è stato creato o aggiornato con successo.")
    print("L'indice è ora configurato per vettori a 384 dimensioni.")
else:
    # Se qualcosa è andato storto, stampa il codice di stato dell'errore e il messaggio di errore dettagliato da Azure.
    print(f"\nERRORE: Azure ha risposto con il codice di stato: {response.status_code}")
    print("Dettagli dell'errore:")
    print(response.json())