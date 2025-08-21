# test_azure.py

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv

print("--- INIZIO TEST CONNESSIONE AZURE ---")

# Carica le variabili d'ambiente
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
    print("❌ ERRORE: Assicurati che AZURE_SEARCH_ENDPOINT e AZURE_SEARCH_API_KEY siano nel file .env")
else:
    print(f"✅ Endpoint trovato: {AZURE_SEARCH_ENDPOINT}")
    try:
        # Prova a creare un client per gli indici
        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )

        # Prova a elencare gli indici presenti nel servizio
        nomi_indici = index_client.list_index_names()
        print("✅ Connessione ad Azure AI Search RIUSCITA!")
        print("Indici trovati nel tuo servizio:")
        for nome in nomi_indici:
            print(f"  - {nome}")

    except Exception as e:
        print(f"❌ ERRORE DI CONNESSIONE AD AZURE: {e}")
        print("\n   Suggerimento: Controlla che la API KEY e l'ENDPOINT siano copiati correttamente dal portale Azure.")

print("--- FINE TEST CONNESSIONE AZURE ---")