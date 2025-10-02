# 02_popola_indice.py
# Questo script gestisce la pipeline di ingestione dei dati per il sistema RAG.
# Si connette a un container di Azure Blob Storage, legge i documenti sorgente (PDF e TXT),
# elabora il testo dividendolo in chunk, genera embedding vettoriali per ogni chunk,
# e carica i dati strutturati nell'indice di Azure AI Search.

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

# Carica le variabili d'ambiente.
load_dotenv()

# Recupera le credenziali.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# Carica il modello di embedding locale.
print("🔎 Caricamento del modello di embedding locale...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello di embedding locale caricato.")

# Inizializza i client dei servizi Azure.
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_API_KEY))
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# --- Funzioni di Elaborazione del Testo ---

def read_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """
    Estrae tutto il testo da un file PDF fornito come stream di byte.
    
    Args:
        stream (io.BytesIO): Lo stream di byte del file PDF.
        
    Returns:
        str: Il contenuto testuale concatenato di tutte le pagine del PDF.
    """
    text = ""
    reader = PyPDF2.PdfReader(stream)
    for page in reader.pages:
        # Estrae il testo da ogni pagina e lo aggiunge. Usa 'or ""' per gestire elegantemente le pagine vuote.
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Suddivide un testo lungo in chunk più piccoli e gestibili.
    Questo è un passo critico in RAG per garantire che gli embedding siano focalizzati e il recupero sia preciso.
    
    Args:
        text (str): Il contenuto testuale completo da suddividere.
        max_chunk_size (int): La lunghezza massima in caratteri per ogni chunk.
        
    Returns:
        list[str]: Una lista di chunk di testo.
    """
    print("Suddivisione del testo in chunk...")
    # Per prima cosa, normalizza gli spazi bianchi per semplificare la suddivisione.
    text = re.sub(r'\s+', ' ', text).strip()
    # Suddivide il testo prima per paragrafi, assumendo che siano unità semantiche significative.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        # Se un paragrafo è più grande della dimensione massima, lo suddivide ulteriormente.
        if len(paragraph) > max_chunk_size:
            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i + max_chunk_size])
        else:
            chunks.append(paragraph)
    print(f"Testo suddiviso in {len(chunks)} chunk.")
    return chunks

# --- Logica Principale ---
# Questo blocco viene eseguito quando lo script è lanciato direttamente.
if __name__ == "__main__":
    print(f"Ricerca di documenti nel container '{AZURE_STORAGE_CONTAINER_NAME}'...")
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    blob_list = container_client.list_blobs()
    
    # Questa lista conterrà tutti i documenti strutturati da caricare in un unico batch.
    documents_to_upload = []

    # Itera attraverso ogni file (blob) trovato nel container di Azure Blob Storage.
    for blob in blob_list:
        print(f"\nElaborazione del file: {blob.name}...")
        blob_client = container_client.get_blob_client(blob.name)
        # Scarica il contenuto del file in uno stream di byte in memoria.
        stream = io.BytesIO(blob_client.download_blob().readall())
        document_text = ""
        
        # --- Gestione del Tipo di File ---
        # Elabora i file in base alla loro estensione.
        if blob.name.lower().endswith('.pdf'):
            document_text = read_text_from_pdf_stream(stream)
        elif blob.name.lower().endswith('.txt'):
            # Per i file di testo, prova a decodificare con UTF-8. Se fallisce, ripiega su
            # una codifica più permissiva come 'latin-1' per evitare di bloccare lo script.
            try:
                document_text = stream.read().decode('utf-8')
            except UnicodeDecodeError:
                print("  - Attenzione: decodifica UTF-8 fallita. Riprovo con 'latin-1' e sostituisco gli errori.")
                stream.seek(0)  # Resetta il puntatore dello stream all'inizio dopo il tentativo di lettura fallito.
                document_text = stream.read().decode('latin-1', errors='replace')
        
        # Salta il file se non è stato possibile estrarre alcun testo.
        if not document_text.strip():
            print(f"Attenzione: non è stato possibile estrarre testo da {blob.name}. Salto il file.")
            continue

        # --- Chunking e Embedding ---
        chunks = split_text_into_chunks(document_text)
        
        print(f"Generazione degli embedding per {len(chunks)} chunk da {blob.name}...")
        for i, chunk in enumerate(chunks):
            # Genera l'embedding vettoriale per il chunk di testo corrente.
            embedding = embedding_model.encode(chunk).tolist()

            # Crea il payload del documento strutturato che corrisponde allo schema dell'indice di Azure AI Search.
            document = {
                "id": str(uuid.uuid4()),      # Un ID unico per ogni chunk.
                "content": chunk,             # Il contenuto testuale.
                "content_vector": embedding   # L'embedding vettoriale.
            }
            documents_to_upload.append(document)

    # --- Caricamento in Batch su Azure AI Search ---
    # Dopo aver elaborato tutti i file, carica la lista di documenti sull'indice in un unico batch.
    if documents_to_upload:
        print(f"\nCaricamento di {len(documents_to_upload)} documenti sull'indice di Azure AI Search...")
        try:
            result = search_client.upload_documents(documents=documents_to_upload)
            # L'oggetto result contiene lo stato per ogni documento caricato.
            print(f"Caricamento completato! {len(result)} documenti indicizzati.")
        except Exception as e:
            print(f"ERRORE durante il caricamento su Azure: {e}")
    else:
        print("Attenzione: nessun documento da caricare è stato trovato.")