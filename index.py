# index.py (Versione aggiornata con embedding LOCALE)

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- CONFIGURAZIONE DEI SERVIZI ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# --- CARICAMENTO DEL MODELLO DI EMBEDDING LOCALE ---
# Usiamo lo stesso modello che usi per la valutazione per coerenza
print("🔎 Caricamento del modello di embedding locale per la ricerca...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello di embedding locale caricato.")

# Configura il client di Azure AI Search
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

def find_products(context: str, top: int = 6) -> list[dict]:
    """
    Usa un modello locale (SentenceTransformer) per vettorizzare la domanda
    e poi interroga Azure AI Search per recuperare i chunk di testo pertinenti.
    """
    # 1. Crea l'embedding della domanda in locale, senza usare Gemini
    query_vector = embedding_model.encode(context).tolist()

    # 2. Esegui la ricerca vettoriale su Azure
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content"]
    )
    
    contesto_recuperato = [result for result in results]
    print(f"✅ Contesto recuperato: {len(contesto_recuperato)} chunk trovati.")
    return contesto_recuperato

# Creiamo un oggetto "product" per coerenza con gli altri script
class ProductFinder:
    def find_products(self, context: str, top: int = 6):
        return find_products(context, top)

product = ProductFinder()