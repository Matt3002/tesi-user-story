# test_singola_domanda.py
# This is a utility script for performing a quick, manual test of the RAG pipeline
# with a single, user-defined question. It allows for rapid testing and debugging
# of different LLM configurations without running the full evaluation suite.

import os
import sys
from dotenv import load_dotenv

# Add the project's root directory to the Python path.
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

# Import the project's custom modules.
from index import product
from writer import  writer_ollama, writer_azure_openai

# Load environment variables from the .env file.
load_dotenv()

# --- Configuration ---
# Set this variable to choose which model to test.
# Available options: "azure_openai", "ollama"
MODELLO_DA_USARE = "azure_openai"
# ---

# 1. User Input Question
domanda_utente = "La ricerca deve funzionare meglio." # Example question

# --- Orchestration Flow ---
if __name__ == "__main__":
    print(f"--- RUNNING SINGLE TEST WITH MODEL: {MODELLO_DA_USARE.upper()} ---")

    # 2. Retrieval: Get context from Azure AI Search.
    print(f"Question: '{domanda_utente}'")
    print("   Retrieving context from Azure AI Search...")
    documenti_trovati = product.find_products(context=domanda_utente)

    # 3. Augmentation: Prepare the context for the prompt.
    if not documenti_trovati:
        print("Warning: No relevant documents found.")
        contesto_per_prompt = "No specific context was found in the knowledge base."
    else:
        contesto_per_prompt = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documenti_trovati]
        )
        print(f"Context prepared from {len(documenti_trovati)} documents.")

    # 4. Generation: Select and call the appropriate writer.
    writers = {
        "ollama": writer_ollama.write,
        "azure_openai": writer_azure_openai.write,
    }

    writer_selezionato = writers.get(MODELLO_DA_USARE)

    if not writer_selezionato:
        print(f"ERROR: Model '{MODELLO_DA_USARE}' is not valid. Please check the configuration.")
        sys.exit()

    print(f"Generating response with {MODELLO_DA_USARE.upper()}...")
    risposta_finale = writer_selezionato(
        productContext=contesto_per_prompt,
        assignment=domanda_utente
    )

    # 5. Output: Print the final result.
    print("\n" + "="*25 + " GENERATED RESULT " + "="*25)
    print(risposta_finale.strip())
    print("="*72)
