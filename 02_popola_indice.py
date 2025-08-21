# populate_from_blob.py (Versione con correzione finale)

import os
import re
import uuid
import io
import google.generativeai as genai
import PyPDF2
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# --- 1. CONFIGURAZIONE DEI SERVIZI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = "user-stories-docs"

# Controlla che tutte le chiavi siano presenti
if not all([GEMINI_API_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME, AZURE_STORAGE_CONNECTION_STRING]):
    raise ValueError("Una o più variabili d'ambiente non sono state impostate. Controlla il file .env")

# Inizializza i client
genai.configure(api_key=GEMINI_API_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# --- 2. FUNZIONI DI ELABORAZIONE ---

def leggi_testo_da_pdf_stream(stream) -> str:
    testo_completo = ""
    reader = PyPDF2.PdfReader(stream)
    for page in reader.pages:
        testo_completo += page.extract_text() or ""
    return testo_completo

def dividi_testo_in_chunk(testo: str, max_chunk_size=1000) -> list[str]:
    print("🔪 Divido il testo in chunk...")
    testo = re.sub(r'\s+', ' ', testo).strip()
    paragrafi = [p.strip() for p in testo.split('\n\n') if p.strip()]
    chunks = []
    for paragrafo in paragrafi:
        if len(paragrafo) > max_chunk_size:
            for i in range(0, len(paragrafo), max_chunk_size):
                chunks.append(paragrafo[i:i + max_chunk_size])
        else:
            chunks.append(paragrafo)
    print(f"✅ Testo diviso in {len(chunks)} chunk.")
    return chunks

# --- 3. LOGICA PRINCIPALE ---

if __name__ == "__main__":
    print(f"🔎 Cerco i documenti nel container '{AZURE_STORAGE_CONTAINER_NAME}'...")
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    blob_list = container_client.list_blobs()
    
    documenti_per_azure = []

    for blob in blob_list:
        print(f"\n📄 Trovato il file: {blob.name}. Elaborazione in corso...")
        blob_client = container_client.get_blob_client(blob.name)
        stream = io.BytesIO(blob_client.download_blob().readall())
        testo_documento = ""
        
        if blob.name.lower().endswith('.pdf'):
            print("  - Rilevato file PDF. Estraggo il testo...")
            testo_documento = leggi_testo_da_pdf_stream(stream)
        elif blob.name.lower().endswith('.txt'):
            print("  - Rilevato file TXT. Leggo il contenuto...")
            testo_documento = stream.read().decode('utf-8')
        
        if not testo_documento.strip():
            print(f"⚠️ Non è stato possibile estrarre testo dal file {blob.name} o il file è vuoto. Salto al prossimo.")
            continue

        chunks = dividi_testo_in_chunk(testo_documento)
        
        print(f"🚀 Genero i vettori (embedding) per i {len(chunks)} chunk del file {blob.name}...")
        for i, chunk in enumerate(chunks):
            print(f"  - Embedding per il chunk {i+1}/{len(chunks)}...")
            embedding_response = genai.embed_content(
                model='models/embedding-001',
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )
            documento = {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "content_vector": embedding_response['embedding']
            }
            documenti_per_azure.append(documento)

    if documenti_per_azure:
        print("\n☁️  Carico tutti i documenti sull'indice di Azure AI Search...")
        try:
            result = search_client.upload_documents(documents=documenti_per_azure)
            # --- ECCO LA CORREZIONE ---
            # La variabile 'result' è una lista, quindi contiamo direttamente la sua lunghezza.
            print(f"✅ Caricamento completato! {len(result)} documenti totali indicizzati.")
            print("\n🎉 L'indice è stato popolato! Ora puoi eseguire hybrid_app.py.")
        except Exception as e:
            print(f"❌ ERRORE durante il caricamento su Azure: {e}")
    else:
        print("⚠️ Nessun documento da caricare. Assicurati che i file contengano testo valido.")