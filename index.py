# index.py
# This module handles the "Retrieval" part of the RAG pipeline.
# Its primary responsibility is to take a user query, transform it into a vector embedding
# using a local SentenceTransformer model, and then query the Azure AI Search index
# to find the most semantically similar text chunks from the knowledge base.

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# --- CONFIGURAZIONE DEL SERVIZIO ---
# Recupera le credenziali e gli identificatori di Azure AI Search dalle variabili d'ambiente.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

#--- CARICAMENTO DEL MODELLO DI EMBEDDING LOCALE ---
# Carica il modello SentenceTransformer direttamente in memoria.
# Questo stesso modello viene utilizzato sia per l'indicizzazione dei documenti (in 02_popola_indice.py)
# sia per le query, il che è cruciale per garantire che la query e i documenti esistano
# nello stesso spazio vettoriale.

print("🔎 Caricamento del modello di embedding locale per la ricerca...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello di embedding locale caricato.")

# Inizializza il client per Azure AI Search.
# Questo oggetto client gestirà tutte le comunicazioni con il servizio di ricerca.
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

def find_products(context: str, top: int = 6) -> list[dict]:
    """
    Performs a semantic search on the Azure AI Search index.

    This function takes a text query, converts it to a vector embedding locally,
    and then sends that vector to Azure to find the 'top' k-nearest neighbors
    (i.e., the most relevant text chunks).

    Args:
        context (str): The user query (or rewritten query) to search for.
        top (int): The number of top results to retrieve. Defaults to 6.

    Returns:
        list[dict]: A list of the search result documents.
    """
    # 1. Vettorizza la Query: Converte il testo di input 'context' in un vettore
    #a 384 dimensioni usando il modello di embedding caricato localmente.
    query_vector = embedding_model.encode(context).tolist()

    # 2. Costruisci la Query Vettoriale: Costruisce un oggetto di query di ricerca che
    #Azure AI Search comprende. Questo specifica il vettore da cercare, quanti vicini
    #trovare (k) e contro quale campo vettoriale nell'indice cercare ('content_vector').
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top,
        fields="content_vector"
    )

    # 3. Esegui la Ricerca: Invia la query al servizio Azure AI Search.
#    'search_text' è None perché stiamo eseguendo una ricerca puramente vettoriale.
#    'select' specifica quali campi restituire dai documenti trovati (qui, solo il contenuto testuale).
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content"]
    )
    
    # Convert the iterator of results into a list.
    contesto_recuperato = [result for result in results]
    print(f"✅ Contesto recuperato: {len(contesto_recuperato)} chunk trovati.")
    return contesto_recuperato

# Questa classe agisce come un semplice wrapper attorno alla funzione find_products.
# È stata creata per fornire un'interfaccia orientata agli oggetti coerente (`product.find_products`)
# che viene utilizzata in altri script come il chatbot e lo script di valutazione.
class ProductFinder:
    def find_products(self, context: str, top: int = 6):
        return find_products(context, top)

# Istanzia il finder in modo che altri moduli possano importarlo e usarlo direttamente.
product = ProductFinder()