# query_rewriter.py
# This module is dedicated to the Query Transformation step of the RAG pipeline.
# Its primary purpose is to take a potentially vague or brief user query and
# rewrite it into a more detailed, semantically rich query that is optimized
# for vector search against a technical knowledge base.

import os
import ollama
from openai import AzureOpenAI

def rewrite_query(user_query: str, model_name: str) -> str:
    """
    Uses a Large Language Model (LLM) to expand and enhance a user's query for better retrieval results.

    Args:
        user_query (str): The original query from the user.
        model_name (str): The identifier for the LLM to use for rewriting (e.g., "azure" or "ollama").

    Returns:
        str: The rewritten, optimized query for semantic search. If an error occurs,
             it returns the original user query as a fallback.
    """
    # Keep the console output in Italian for the user.
    print(f"   - Riformulo la query originale: '{user_query}'...")

    # This is a "meta-prompt" designed specifically for the rewriting task.
    # It instructs the LLM to act as an information retrieval expert and to only
    # rephrase the query, not answer it. This ensures the output is clean and focused.
    system_prompt_for_rewriting = (
        "Sei un assistente AI esperto in information retrieval. Il tuo unico compito è prendere una richiesta utente, "
        "spesso breve o ambigua, e trasformarla in una query di ricerca dettagliata e semanticamente ricca. "
        "La query ottimizzata deve essere ideale per una ricerca vettoriale in una base di conoscenza tecnica. "
        "NON rispondere alla domanda, ma RIFORMULALA.\n"
        "Restituisci solo ed esclusivamente la query migliorata, senza alcuna frase introduttiva o di contorno."
    )
    
    # The user prompt combines the static instruction with the dynamic user query.
    user_prompt_for_rewriting = f"Riscrivi e ottimizza la seguente richiesta per una ricerca semantica: \"{user_query}\""
    
    # A robust try-except block handles potential API errors (e.g., network issues, invalid keys).
    # If any error occurs, the function will fall back to using the original query,
    # ensuring the RAG pipeline does not crash.
    try:
        rewritten_query = ""
        # --- Model Selection Logic ---
        # Selects the appropriate LLM service based on the 'model_name' parameter.

        if model_name == "azure":
            # Initialize the Azure OpenAI client using credentials from environment variables.
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            # Send the request to the Azure OpenAI API with the specialized rewriting prompts.
            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"), # Reads the deployment name from .env
                messages=[
                    {"role": "system", "content": system_prompt_for_rewriting},
                    {"role": "user", "content": user_prompt_for_rewriting}
                ]
            )
            rewritten_query = response.choices[0].message.content

        elif model_name == "ollama":
            # Read the local model name from .env, with a fallback to a default value.
            ollama_model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:latest")
            print(f"   - Tento la riformulazione con il modello Ollama: '{ollama_model}'")
            
            # Send the request to the local Ollama service.
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt_for_rewriting},
                    {'role': 'user', 'content': user_prompt_for_rewriting},
                ]
            )
            rewritten_query = response['message']['content']
            
        else:
            # If the model_name is not recognized, skip rewriting and return the original query.
            print(f"   - ATTENZIONE: Modello '{model_name}' non valido. Uso la query originale.")
            return user_query

        # --- Final Processing ---
        # Clean up the LLM's response by removing leading/trailing whitespace and any quotation marks
        # to ensure a clean query is passed to the search index.
        final_query = rewritten_query.strip().replace('"', '')
        print(f"   - Query ottimizzata: '{final_query}'")
        return final_query

    except Exception as e:
        # If any exception occurs during the API call, log the error and return the original query.
        print(f"   - ERRORE durante la riformulazione della query: {e}. Verrà usata la query originale.")
        return user_query