# 02_popola_indice.py
# This script handles the data ingestion pipeline for the RAG system.
# It connects to an Azure Blob Storage container, reads source documents (PDFs and TXTs),
# processes the text, generates vector embeddings, and uploads the structured data
# to the Azure AI Search index created by the '01_crea_indice.py' script.

import os
import re
import uuid
import io
import PyPDF2
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from the .env file.
load_dotenv()

# --- Service Configuration ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = "user-stories-docs"

# Validate that all required environment variables are set.
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME, AZURE_STORAGE_CONNECTION_STRING]):
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# --- Local Embedding Model Loading ---
# This model converts text chunks into numerical vectors (embeddings).
# It's crucial to use the same model here as in the retrieval script ('index.py').
print("Loading local embedding model for data population...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Local embedding model loaded successfully.")

# Initialize Azure service clients.
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_API_KEY))
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# --- Text Processing Functions ---

def read_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """Extracts all text from a PDF file stream."""
    text = ""
    reader = PyPDF2.PdfReader(stream)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Splits a long text into smaller chunks.
    Chunking is essential for RAG because it creates focused, semantically coherent
    units for embedding, leading to more precise retrieval.
    """
    print("Splitting text into chunks...")
    # Normalize whitespace.
    text = re.sub(r'\s+', ' ', text).strip()
    # A simple strategy: split by paragraphs first.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            # If a paragraph is too long, split it further.
            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i + max_chunk_size])
        else:
            chunks.append(paragraph)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# --- Main Logic ---
if __name__ == "__main__":
    print(f"Searching for documents in container '{AZURE_STORAGE_CONTAINER_NAME}'...")
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    blob_list = container_client.list_blobs()
    
    documents_to_upload = []

    for blob in blob_list:
        print(f"\nProcessing file: {blob.name}...")
        blob_client = container_client.get_blob_client(blob.name)
        stream = io.BytesIO(blob_client.download_blob().readall())
        document_text = ""
        
        # Extract text based on file extension.
        if blob.name.lower().endswith('.pdf'):
            document_text = read_text_from_pdf_stream(stream)
        elif blob.name.lower().endswith('.txt'):
            document_text = stream.read().decode('utf-8')
        
        if not document_text.strip():
            print(f"Warning: No text could be extracted from {blob.name}. Skipping.")
            continue

        chunks = split_text_into_chunks(document_text)
        
        print(f"Generating embeddings for {len(chunks)} chunks from {blob.name}...")
        for i, chunk in enumerate(chunks):
            # Generate the vector embedding for the current chunk.
            embedding = embedding_model.encode(chunk).tolist()

            # Create the document structure that matches the Azure AI Search index schema.
            document = {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "content_vector": embedding
            }
            documents_to_upload.append(document)

    # Upload all prepared documents to the index in a single batch request.
    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to the Azure AI Search index...")
        try:
            result = search_client.upload_documents(documents=documents_to_upload)
            print(f"Upload complete! {len(result)} documents indexed.")
        except Exception as e:
            print(f"ERROR during upload to Azure: {e}")
    else:
        print("Warning: No documents were found to upload.")
