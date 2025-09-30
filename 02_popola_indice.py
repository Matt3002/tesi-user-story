# 02_popola_indice.py
# This script handles the data ingestion pipeline for the RAG system.
# It connects to an Azure Blob Storage container, reads source documents (PDFs and TXTs),
# processes the text by splitting it into chunks, generates vector embeddings for each chunk,
# and uploads the structured data to the Azure AI Search index.

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
# Retrieve credentials and identifiers for Azure services from environment variables.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = "user-stories-docs" # The specific container holding the knowledge base documents.

# Validate that all required environment variables are set to prevent runtime errors.
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME, AZURE_STORAGE_CONNECTION_STRING]):
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# --- Local Embedding Model Loading ---
# Load the SentenceTransformer model that will be used to convert text chunks into vector embeddings.
# It's crucial to use the same model here as in the querying script (index.py) to ensure
# vectors are in the same dimensional space.
print("Loading local embedding model for data population...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Local embedding model loaded successfully.")

# Initialize Azure service clients.
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_API_KEY))
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# --- Text Processing Functions ---

def read_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """
    Extracts all text from a PDF file provided as a byte stream.
    
    Args:
        stream (io.BytesIO): The byte stream of the PDF file.
        
    Returns:
        str: The concatenated text content from all pages of the PDF.
    """
    text = ""
    reader = PyPDF2.PdfReader(stream)
    for page in reader.pages:
        # Extract text from each page and append it. Use 'or ""' to handle empty pages gracefully.
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Splits a long text into smaller, manageable chunks.
    This is a critical step in RAG to ensure that embeddings are focused and retrieval is precise.
    
    Args:
        text (str): The full text content to be split.
        max_chunk_size (int): The maximum character length for each chunk.
        
    Returns:
        list[str]: A list of text chunks.
    """
    print("Splitting text into chunks...")
    # First, normalize whitespace to simplify splitting.
    text = re.sub(r'\s+', ' ', text).strip()
    # Split the text by paragraphs first, assuming they are meaningful semantic units.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        # If a paragraph is larger than the max size, split it further.
        if len(paragraph) > max_chunk_size:
            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i + max_chunk_size])
        else:
            chunks.append(paragraph)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# --- Main Logic ---
# This block executes when the script is run directly.
if __name__ == "__main__":
    print(f"Searching for documents in container '{AZURE_STORAGE_CONTAINER_NAME}'...")
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    blob_list = container_client.list_blobs()
    
    # This list will hold all the structured documents to be uploaded in a single batch.
    documents_to_upload = []

    # Iterate through each file (blob) found in the Azure Blob Storage container.
    for blob in blob_list:
        print(f"\nProcessing file: {blob.name}...")
        blob_client = container_client.get_blob_client(blob.name)
        # Download the file content into an in-memory byte stream.
        stream = io.BytesIO(blob_client.download_blob().readall())
        document_text = ""
        
        # --- File Type Handling ---
        # Process files based on their extension.
        if blob.name.lower().endswith('.pdf'):
            document_text = read_text_from_pdf_stream(stream)
        elif blob.name.lower().endswith('.txt'):
            # For text files, try decoding with UTF-8. If it fails, fall back to
            # a more permissive encoding like 'latin-1' to avoid crashing the script.
            try:
                document_text = stream.read().decode('utf-8')
            except UnicodeDecodeError:
                print("   - Warning: UTF-8 decoding failed. Retrying with 'latin-1' and replacing errors.")
                stream.seek(0)  # Reset stream pointer to the beginning after the failed read attempt.
                document_text = stream.read().decode('latin-1', errors='replace')
        
        # Skip the file if no text could be extracted.
        if not document_text.strip():
            print(f"Warning: No text could be extracted from {blob.name}. Skipping.")
            continue

        # --- Chunking and Embedding ---
        chunks = split_text_into_chunks(document_text)
        
        print(f"Generating embeddings for {len(chunks)} chunks from {blob.name}...")
        for i, chunk in enumerate(chunks):
            # Generate the vector embedding for the current text chunk.
            embedding = embedding_model.encode(chunk).tolist()

            # Create the structured document payload that matches the Azure AI Search index schema.
            document = {
                "id": str(uuid.uuid4()),      # A unique ID for each chunk.
                "content": chunk,             # The text content.
                "content_vector": embedding   # The vector embedding.
            }
            documents_to_upload.append(document)

    # --- Batch Upload to Azure AI Search ---
    # After processing all files, upload the list of documents to the index in one batch.
    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to the Azure AI Search index...")
        try:
            result = search_client.upload_documents(documents=documents_to_upload)
            # The result object contains the status for each uploaded document.
            print(f"Upload complete! {len(result)} documents indexed.")
        except Exception as e:
            print(f"ERROR during upload to Azure: {e}")
    else:
        print("Warning: No documents were found to upload.")