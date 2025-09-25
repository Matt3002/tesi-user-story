# chatbot.py
# This script provides an interactive command-line interface (CLI) to chat
# with the RAG-powered User Story generation agent. It has been updated to
# allow the user to choose between different generation models (Azure or Ollama)
# at startup, and to be run in a non-interactive mode for automation.

import os
import sys
import argparse  # Import the argument parser library
from dotenv import load_dotenv

# --- Path Setup ---
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

# --- Module Imports ---
from index import product
from writer import writer_azure_openai, writer_ollama

# --- Configuration ---
load_dotenv()

# --- Core RAG Logic Function ---
def generate_user_story(user_query: str, model_name: str) -> str:
    """
    Executes the full RAG pipeline for a given user query using the selected model.
    """
    print(f"\n...Recupero contesto per la query: '{user_query}'...")
    
    documents = product.find_products(context=user_query, top=6)
    
    if not documents:
        context_for_prompt = "Nessun contesto specifico trovato nella base di conoscenza."
        print("⚠️  Nessun contesto pertinente trovato.")
    else:
        context_for_prompt = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documents]
        )
        print(f"✅  Contesto basato su {len(documents)} documenti recuperato.")

    writers = {
        "azure": writer_azure_openai.write,
        "ollama": writer_ollama.write
    }
    
    writer_function = writers.get(model_name)
    
    if not writer_function:
        return f"Errore: Modello '{model_name}' non riconosciuto."
        
    print(f"...Genero la user story con {model_name.upper()}...")
    
    final_response = writer_function(
        productContext=context_for_prompt,
        assignment=user_query
    )
    
    return final_response

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up an argument parser to handle non-interactive execution for GitHub Actions.
    parser = argparse.ArgumentParser(description="Agente Chatbot per la generazione di User Story.")
    parser.add_argument('--model', type=str, choices=['azure', 'ollama'], help="Il modello da utilizzare ('azure' o 'ollama').")
    parser.add_argument('--query', type=str, help="La richiesta per la user story (per l'esecuzione non interattiva).")
    args = parser.parse_args()

    # --- NON-INTERACTIVE MODE ---
    # If the script is run with command-line arguments, execute once and exit.
    if args.model and args.query:
        print(f"✅ Esecuzione non interattiva con il modello: {args.model.upper()}")
        user_story = generate_user_story(args.query, args.model)
        print("\n--- User Story Generata ---")
        print(user_story.strip())
        print("---------------------------\n")
    
    # --- INTERACTIVE MODE ---
    # If no arguments are provided, start the interactive chat session.
    else:
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
        
        while True:
            user_input = input("\n> La tua richiesta: ")
            
            if user_input.lower() in ['esci', 'exit', 'quit']:
                print("Arrivederci!")
                break
                
            user_story = generate_user_story(user_input, selected_model)
            
            print("\n--- User Story Generata ---")
            print(user_story.strip())
            print("---------------------------\n")

