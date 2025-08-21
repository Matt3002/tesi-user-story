# test_gemini.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

print("--- INIZIO TEST CONNESSIONE GEMINI ---")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERRORE: Assicurati che GEMINI_API_KEY sia impostata nel file .env")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # Prova a elencare i modelli disponibili
        print("✅ Chiave API trovata. Provo a contattare i server di Google...")
        model_list = [m.name for m in genai.list_models()]

        if 'models/embedding-001' in model_list:
             print("✅ Connessione a Gemini RIUSCITA!")
             print("   Il modello 'embedding-001' è disponibile.")
        else:
             print("⚠️ Connessione riuscita, ma il modello 'embedding-001' non è stato trovato.")

    except Exception as e:
        print(f"❌ ERRORE DI CONNESSIONE A GEMINI: {e}")
        print("\n   Suggerimento: La chiave API potrebbe essere errata, scaduta o non avere il servizio 'Generative Language API' abilitato nel tuo progetto Google Cloud.")

print("--- FINE TEST CONNESSIONE GEMINI ---")