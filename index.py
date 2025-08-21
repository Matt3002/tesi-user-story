# index.py

import os
import google.generativeai as genai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURAZIONE DEI SERVIZI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configura i client
genai.configure(api_key=GEMINI_API_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

def find_products(context: str, top: int = 3) -> list[dict]:
    """
    Usa l'API di Gemini per vettorizzare la domanda e poi interroga
    Azure AI Search per recuperare i chunk di testo pertinenti.
    """
    embedding_response = genai.embed_content(
        model='models/embedding-001',
        content=context,
        task_type="RETRIEVAL_QUERY"
    )
    query_vector = embedding_response['embedding']

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

# Creiamo un oggetto "product" per coerenza con il tuo import
# In questo modo, l'import "from index import product" funzionerà
class ProductFinder:
    def find_products(self, context: str, top: int = 3):
        return find_products(context, top)

product = ProductFinder()