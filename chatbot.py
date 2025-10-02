# chatbot.py
# Questo script fornisce un'interfaccia a riga di comando (CLI) interattiva per chattare
# con l'agente di generazione di User Story basato su RAG. È stato aggiornato per
# includere il passo di riscrittura della query per migliorare la qualità del recupero, permettere
# all'utente di scegliere tra i modelli all'avvio ed essere eseguito in modalità non interattiva per l'automazione.

import os
import sys
import argparse
from dotenv import load_dotenv

# --- Setup del Percorso ---
# Aggiunge la directory principale del progetto al path di Python per consentire l'importazione di moduli locali.
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

# --- Importazione dei Moduli ---
# Importa i moduli necessari dalla struttura del progetto.
from index import product                           # Gestisce il passo di Recupero (Retrieval)
from writer import writer_azure_openai, writer_ollama # Gestiscono il passo di Generazione (Generation)
from query_rewriter import rewrite_query              # Gestisce il passo di Trasformazione della Query

# --- Configurazione ---
# Carica le variabili d'ambiente (chiavi API, endpoint) dal file .env.
load_dotenv()

# --- Funzione Logica Principale del RAG ---
def generate_user_story(user_query: str, model_name: str) -> str:
    """
    Esegue l'intera pipeline RAG per una data query dell'utente usando il modello selezionato.
    Questa funzione ora include il passo di riscrittura della query.

    Args:
        user_query (str): La query originale inserita dall'utente.
        model_name (str): L'identificatore per il modello da usare ("azure" o "ollama").

    Returns:
        str: La user story finale generata.
    """
    # 1. Riscrittura della Query: Per prima cosa, riscrive la query dell'utente per renderla più efficace per la ricerca semantica.
    # L'output a console per questo passo è gestito all'interno della funzione rewrite_query.
    rewritten_search_query = rewrite_query(user_query, model_name)
    
    # Mantiene l'output a console rivolto all'utente in italiano.
    print(f"\n...Recupero contesto per la query: '{rewritten_search_query}'...")
    
    # 2. Recupero: Usa la query riscritta per trovare i 6 chunk di documenti più pertinenti.
    documents = product.find_products(context=rewritten_search_query, top=6)
    
    # 3. Aumento: Prepara il contesto recuperato per il prompt finale.
    if not documents:
        context_for_prompt = "Nessun contesto specifico trovato nella base di conoscenza."
        print("⚠️  Nessun contesto pertinente trovato.")
    else:
        context_for_prompt = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documents]
        )
        print(f"✅  Contesto basato su {len(documents)} documenti recuperato.")

    # 4. Generazione: Seleziona la funzione di scrittura appropriata in base al modello scelto.
    writers = {
        "azure": writer_azure_openai.write,
        "ollama": writer_ollama.write
    }
    
    writer_function = writers.get(model_name)
    
    if not writer_function:
        return f"Errore: Modello '{model_name}' non riconosciuto."
        
    print(f"...Genero la user story con {model_name.upper()}...")
    
    # La chiamata finale allo scrittore usa il contesto aumentato ma la query ORIGINALE dell'utente
    # per garantire che la storia generata risponda direttamente alla richiesta iniziale.
    final_response = writer_function(
        productContext=context_for_prompt,
        assignment=user_query
    )
    
    return final_response

# --- Blocco di Esecuzione Principale ---
# Questo blocco viene eseguito quando lo script è lanciato direttamente.
if __name__ == "__main__":
    # Imposta un parser di argomenti per gestire sia l'esecuzione interattiva che non interattiva.
    # Questo è utile per flussi di lavoro automatizzati come le GitHub Actions.
    parser = argparse.ArgumentParser(description="Agente Chatbot per la generazione di User Story.")
    parser.add_argument('--model', type=str, choices=['azure', 'ollama'], help="Il modello da utilizzare ('azure' o 'ollama').")
    parser.add_argument('--query', type=str, help="La richiesta per la user story (per l'esecuzione non interattiva).")
    args = parser.parse_args()

    # --- MODALITÀ NON INTERATTIVA ---
    # Se lo script viene eseguito con gli argomenti '--model' e '--query', esegue una volta e termina.
    if args.model and args.query:
        print(f"✅ Esecuzione non interattiva con il modello: {args.model.upper()}")
        user_story = generate_user_story(args.query, args.model)
        print("\n--- User Story Generata ---")
        print(user_story.strip())
        print("---------------------------\n")
    
    # --- MODALITÀ INTERATTIVA ---
    # Se non vengono forniti argomenti, lo script avvia una sessione di chat interattiva.
    else:
        # I prompt rivolti all'utente sono in italiano.
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
        
        # Avvia un ciclo per accettare continuamente l'input dell'utente.
        while True:
            user_input = input("\n> La tua richiesta: ")
            
            if user_input.lower() in ['esci', 'exit', 'quit']:
                print("Arrivederci!")
                break
            
            # Per ogni input, esegue l'intera pipeline RAG con il riscrittore di query.
            user_story = generate_user_story(user_input, selected_model)
            
            # Stampa il risultato formattato.
            print("\n--- User Story Generata ---")
            print(user_story.strip())
            print("---------------------------\n")