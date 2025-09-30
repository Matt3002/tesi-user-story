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

# Load environment variables from the .env file at the root of the project.
load_dotenv()

# --- SERVICE CONFIGURATION ---
# Retrieve Azure AI Search credentials and identifiers from environment variables.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# --- LOCAL EMBEDDING MODEL LOADING ---
# Load the SentenceTransformer model directly into memory.
# This same model is used for both indexing documents (in 02_popola_indice.py) and for
# querying, which is crucial for ensuring that the query and the documents exist
# in the same vector space.
# The console output is kept in Italian for the user.
print("🔎 Caricamento del modello di embedding locale per la ricerca...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello di embedding locale caricato.")

# Initialize the client for Azure AI Search.
# This client object will handle all communications with the search service.
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
    # 1. Vectorize the Query: Convert the input text 'context' into a 384-dimension
    #    vector using the locally loaded embedding model.
    query_vector = embedding_model.encode(context).tolist()

    # 2. Build the Vector Query: Construct a search query object that Azure AI Search
    #    understands. This specifies the vector to search for, how many neighbors to find (k),
    #    and which vector field in the index to search against ('content_vector').
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top,
        fields="content_vector"
    )

    # 3. Execute the Search: Send the query to the Azure AI Search service.
    #    'search_text' is None because we are performing a pure vector search.
    #    'select' specifies which fields to return from the found documents (here, only the text content).
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content"]
    )
    
    # Convert the iterator of results into a list.
    contesto_recuperato = [result for result in results]
    print(f"✅ Contesto recuperato: {len(contesto_recuperato)} chunk trovati.")
    return contesto_recuperato

# This class acts as a simple wrapper around the find_products function.
# It was created to provide a consistent object-oriented interface (`product.find_products`)
# that is used across other scripts like the chatbot and the evaluation script.
class ProductFinder:
    def find_products(self, context: str, top: int = 6):
        return find_products(context, top)

# Instantiate the finder so other modules can import and use it directly.
product = ProductFinder()