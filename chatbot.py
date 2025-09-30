# chatbot.py
# This script provides an interactive command-line interface (CLI) to chat
# with the RAG-powered User Story generation agent. It has been updated to
# include the query rewriting step to improve retrieval quality, allow the user
# to choose between models at startup, and run in a non-interactive mode for automation.

import os
import sys
import argparse
from dotenv import load_dotenv

# --- Path Setup ---
# Add the project's root directory to the Python path to allow importing local modules.
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

# --- Module Imports ---
# Import the necessary modules from the project structure.
from index import product                          # Handles the Retrieval step
from writer import writer_azure_openai, writer_ollama  # Handle the Generation step
from query_rewriter import rewrite_query           # Handles the Query Transformation step

# --- Configuration ---
# Load environment variables (API keys, endpoints) from the .env file.
load_dotenv()

# --- Core RAG Logic Function ---
def generate_user_story(user_query: str, model_name: str) -> str:
    """
    Executes the full RAG pipeline for a given user query using the selected model.
    This function now includes the query rewriting step.

    Args:
        user_query (str): The original query entered by the user.
        model_name (str): The identifier for the model to use ("azure" or "ollama").

    Returns:
        str: The final, generated user story.
    """
    # 1. Query Rewriting: First, rewrite the user's query to be more effective for semantic search.
    # The console output for this step is handled within the rewrite_query function.
    rewritten_search_query = rewrite_query(user_query, model_name)
    
    # Keep user-facing console output in Italian.
    print(f"\n...Recupero contesto per la query: '{rewritten_search_query}'...")
    
    # 2. Retrieval: Use the rewritten query to find the top 6 most relevant document chunks.
    documents = product.find_products(context=rewritten_search_query, top=6)
    
    # 3. Augmentation: Prepare the retrieved context for the final prompt.
    if not documents:
        context_for_prompt = "Nessun contesto specifico trovato nella base di conoscenza."
        print("⚠️  Nessun contesto pertinente trovato.")
    else:
        context_for_prompt = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documents]
        )
        print(f"✅  Contesto basato su {len(documents)} documenti recuperato.")

    # 4. Generation: Select the appropriate writer function based on the chosen model.
    writers = {
        "azure": writer_azure_openai.write,
        "ollama": writer_ollama.write
    }
    
    writer_function = writers.get(model_name)
    
    if not writer_function:
        return f"Errore: Modello '{model_name}' non riconosciuto."
        
    print(f"...Genero la user story con {model_name.upper()}...")
    
    # The final call to the writer uses the augmented context but the ORIGINAL user query
    # to ensure the generated story directly answers the initial request.
    final_response = writer_function(
        productContext=context_for_prompt,
        assignment=user_query
    )
    
    return final_response

# --- Main Execution Block ---
# This block runs when the script is executed directly.
if __name__ == "__main__":
    # Set up an argument parser to handle both interactive and non-interactive execution.
    # This is useful for automated workflows like GitHub Actions.
    parser = argparse.ArgumentParser(description="Agente Chatbot per la generazione di User Story.")
    parser.add_argument('--model', type=str, choices=['azure', 'ollama'], help="Il modello da utilizzare ('azure' o 'ollama').")
    parser.add_argument('--query', type=str, help="La richiesta per la user story (per l'esecuzione non interattiva).")
    args = parser.parse_args()

    # --- NON-INTERACTIVE MODE ---
    # If the script is run with '--model' and '--query' arguments, it executes once and exits.
    if args.model and args.query:
        print(f"✅ Esecuzione non interattiva con il modello: {args.model.upper()}")
        user_story = generate_user_story(args.query, args.model)
        print("\n--- User Story Generata ---")
        print(user_story.strip())
        print("---------------------------\n")
    
    # --- INTERACTIVE MODE ---
    # If no arguments are provided, the script starts an interactive chat session.
    else:
        # User-facing prompts are in Italian.
        print("Benvenuto nell'Agente per User Story!")
        print("Scegli il modello da utilizzare:")
        print("  1: Azure OpenAI (gpt-4o) - Potente e affidabile (Cloud)")
        print("  2: Ollama (Llama 3.2) - Veloce e sperimentale (Locale)")
        
        model_choice = ""
        while model_choice not in ["1", "2"]:
            model_choice = input("Inserisci la tua scelta (1 o 2): ")

        if model_choice == "1":
            selected_model = "azure"
            print("\n✅ Modello selezionato: Azure OpenAI (gpt-4o).\n")
        else:
            selected_model = "ollama"
            print("\n✅ Modello selezionato: Ollama (Llama 3.2).\n")
        
        print("Scrivi la tua richiesta per una user story. Digita 'esci' per terminare.")
        
        # Start a loop to continuously accept user input.
        while True:
            user_input = input("\n> La tua richiesta: ")
            
            if user_input.lower() in ['esci', 'exit', 'quit']:
                print("Arrivederci!")
                break
            
            # For each input, run the full RAG pipeline with the query rewriter.
            user_story = generate_user_story(user_input, selected_model)
            
            # Print the formatted result.
            print("\n--- User Story Generata ---")
            print(user_story.strip())
            print("---------------------------\n")