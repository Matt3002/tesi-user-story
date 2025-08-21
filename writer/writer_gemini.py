# writers/writer_gemini.py (Versione Finale - Senza PromptFlow)

import google.generativeai as genai
from promptflow.tracing import trace

# Il modello di Gemini che vogliamo usare
GEMINI_MODEL_NAME = 'gemini-1.5-flash'

@trace
def write(productContext: str, assignment: str) -> str:
    """
    Genera una risposta con Gemini usando la chiamata API diretta,
    bypassando la libreria PromptFlow per la chiamata al modello.

    Args:
        productContext (str): Il contesto recuperato da Azure.
        assignment (str): La domanda originale dell'utente.

    Returns:
        str: La risposta generata dal modello Gemini.
    """
    # 1. Costruiamo il prompt manualmente, replicando la logica del file .prompty
    prompt = f"""
system:
Sei un Product Manager esperto nella scrittura di user story Agile.
Usa i seguenti esempi recuperati dalla nostra base di conoscenza per basare la tua risposta.
Non usare informazioni esterne. Se il contesto non è sufficiente, indicalo.

# Contesto recuperato da Azure AI Search
{productContext}

user:
Basandoti SUL CONTESTO FORNITO, genera una user story completa per la seguente richiesta: "{assignment}"
"""

    # 2. Inizializziamo il modello Gemini
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # 3. Generiamo la risposta
    response = model.generate_content(prompt)

    # 4. Restituiamo il testo della risposta
    return response.text