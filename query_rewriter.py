# query_rewriter.py
# Questo modulo è dedicato al passo di Trasformazione della Query nella pipeline RAG.
# Il suo scopo principale è prendere una query dell'utente, potenzialmente vaga o breve, e
# riscriverla in una query più dettagliata e semanticamente ricca, ottimizzata
# per una ricerca vettoriale in una base di conoscenza tecnica.

import os
import ollama
from openai import AzureOpenAI

def rewrite_query(user_query: str, model_name: str) -> str:
    """
    Usa un Large Language Model (LLM) per espandere e migliorare la query di un utente per ottenere risultati di recupero migliori.

    Args:
        user_query (str): La query originale dell'utente.
        model_name (str): L'identificatore per l'LLM da usare per la riscrittura (es. "azure" o "ollama").

    Returns:
        str: La query riscritta e ottimizzata per la ricerca semantica. Se si verifica un errore,
             restituisce la query originale dell'utente come fallback.
    """
    # Mantiene l'output della console in italiano per l'utente.
    print(f"   - Riformulo la query originale: '{user_query}'...")

    # Questo è un "meta-prompt" progettato specificamente per il compito di riscrittura.
    # Istruisce l'LLM ad agire come un esperto di recupero dell'informazione e a
    # riformulare soltanto la query, non a rispondere. Ciò garantisce che l'output sia pulito e mirato.
    system_prompt_for_rewriting = (
        "Sei un assistente AI esperto in information retrieval. Il tuo unico compito è prendere una richiesta utente, "
        "spesso breve o ambigua, e trasformarla in una query di ricerca dettagliata e semanticamente ricca. "
        "La query ottimizzata deve essere ideale per una ricerca vettoriale in una base di conoscenza tecnica. "
        "NON rispondere alla domanda, ma RIFORMULALA.\n"
        "Restituisci solo ed esclusivamente la query migliorata, senza alcuna frase introduttiva o di contorno."
    )
    
    # Il prompt per l'utente combina l'istruzione statica con la query dinamica dell'utente.
    user_prompt_for_rewriting = f"Riscrivi e ottimizza la seguente richiesta per una ricerca semantica: \"{user_query}\""
    
    # Un blocco try-except robusto gestisce potenziali errori dell'API (es. problemi di rete, chiavi non valide).
    # Se si verifica un errore, la funzione ripiegherà sull'uso della query originale,
    # garantendo che la pipeline RAG non si blocchi.
    try:
        rewritten_query = ""
        # --- Logica di Selezione del Modello ---
        # Seleziona il servizio LLM appropriato in base al parametro 'model_name'.

        if model_name == "azure":
            # Inizializza il client di Azure OpenAI usando le credenziali dalle variabili d'ambiente.
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            # Invia la richiesta all'API di Azure OpenAI con i prompt specializzati per la riscrittura.
            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"), # Legge il nome del deployment da .env
                messages=[
                    {"role": "system", "content": system_prompt_for_rewriting},
                    {"role": "user", "content": user_prompt_for_rewriting}
                ]
            )
            rewritten_query = response.choices[0].message.content

        elif model_name == "ollama":
            # Legge il nome del modello locale da .env, con un fallback a un valore predefinito.
            ollama_model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:latest")
            print(f"   - Tento la riformulazione con il modello Ollama: '{ollama_model}'")
            
            # Invia la richiesta al servizio Ollama locale.
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt_for_rewriting},
                    {'role': 'user', 'content': user_prompt_for_rewriting},
                ]
            )
            rewritten_query = response['message']['content']
            
        else:
            # Se il model_name non è riconosciuto, salta la riscrittura e restituisce la query originale.
            print(f"   - ATTENZIONE: Modello '{model_name}' non valido. Uso la query originale.")
            return user_query

        # --- Elaborazione Finale ---
        # Pulisce la risposta dell'LLM rimuovendo spazi bianchi iniziali/finali e qualsiasi virgoletta
        # per garantire che una query pulita venga passata all'indice di ricerca.
        final_query = rewritten_query.strip().replace('"', '')
        print(f"   - Query ottimizzata: '{final_query}'")
        return final_query

    except Exception as e:
        # Se si verifica un'eccezione durante la chiamata API, registra l'errore e restituisce la query originale.
        print(f"   - ERRORE durante la riformulazione della query: {e}. Verrà usata la query originale.")
        return user_query