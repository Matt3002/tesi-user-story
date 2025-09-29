# 03_valuta_modello.py
# This script performs a quantitative evaluation of the RAG system.
# It iterates through a predefined 'golden dataset' of questions and ideal answers,
# runs the full RAG pipeline for each question, and calculates the semantic similarity
# between the generated answer and the ideal answer. Items where the model responds
# with "Context not sufficient" are logged in the report but excluded from the
# final average score calculation for a more accurate performance metric.

import os
import sys
import json
import time
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add the project's root directory to the Python path to allow importing local modules.
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

# Import the project's custom modules.
from index import product
from writer import writer_ollama, writer_azure_openai

# --- Model Configuration ---
MODELLO_DA_USARE = "ollama"  # Available options: "azure_openai", "ollama"
# ---

def salva_report_excel(risultati, punteggio_medio_totale, risultati_cat, num_validi, num_totali):
    """Saves the evaluation results into a well-formatted Excel file."""
    print("\nSaving the report to an Excel file...")
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "RAG Evaluation Report"

        headers = ["Category", "Question", "Ideal Answer", "Generated Answer", "Similarity Score"]
        ws.append(headers)
        
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        excluded_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for item in risultati:
            ws.append([item["categoria"], item["domanda"], item["risposta_ideale"], item["risposta_generata"], f'{item["score"]:.4f}'])
            # Highlight rows that were excluded from the average calculation.
            if "contesto non sufficiente" in item["risposta_generata"].lower():
                for cell in ws[ws.max_row]:
                    cell.fill = excluded_fill

        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        for row in ws.iter_rows(min_row=2, max_row=len(risultati)+1):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                cell.border = thin_border

        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 60
        ws.column_dimensions['C'].width = 60
        ws.column_dimensions['D'].width = 60
        ws.column_dimensions['E'].width = 20
        
        start_row = len(risultati) + 4
        ws.cell(row=start_row, column=1, value="Evaluation Summary").font = Font(bold=True, size=14)
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=2)
        
        ws.cell(row=start_row + 1, column=1, value="Model Used:").font = Font(bold=True)
        ws.cell(row=start_row + 1, column=2, value=MODELLO_DA_USARE.upper())
        
        ws.cell(row=start_row + 2, column=1, value="Evaluated Samples:").font = Font(bold=True)
        ws.cell(row=start_row + 2, column=2, value=f"{num_validi} / {num_totali}")
        
        ws.cell(row=start_row + 3, column=1, value="Overall Average Score:").font = Font(bold=True)
        ws.cell(row=start_row + 3, column=2, value=f"{punteggio_medio_totale:.4f}").font = Font(bold=True, color="00B050")
        
        ws.cell(row=start_row + 5, column=1, value="Average Scores by Category").font = Font(bold=True, size=12)
        ws.merge_cells(start_row=start_row+5, start_column=1, end_row=start_row+5, end_column=2)

        cat_row = start_row + 6
        for categoria, punteggio in risultati_cat.items():
            ws.cell(row=cat_row, column=1, value=f"Category '{categoria}'")
            ws.cell(row=cat_row, column=2, value=f"{punteggio['media']:.4f} ({punteggio['validi']}/{punteggio['totali']} samples)").font = Font(bold=True)
            cat_row += 1
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'report_valutazione_{timestamp}.xlsx'
        wb.save(file_name)
        print(f"Report saved successfully as '{file_name}'")
        
    except Exception as e:
        print(f"ERROR while saving the Excel report: {e}")


def esegui_rag(domanda: str) -> str:
    """Orchestrates the full RAG pipeline for a single question."""
    print(f"   - Running RAG for question: '{domanda}'")
    documenti_trovati = product.find_products(context=domanda)
    contesto_per_prompt = "\n\n---\n\n".join([doc.get("content", "") for doc in documenti_trovati]) if documenti_trovati else "No specific context was found."
    
    writers = {
        "ollama": writer_ollama.write,
        "azure_openai": writer_azure_openai.write
    }
    return writers[MODELLO_DA_USARE](productContext=contesto_per_prompt, assignment=domanda)

# --- Main Evaluation Flow ---
if __name__ == "__main__":
    print(f"--- STARTING EVALUATION (SELECTED MODEL: {MODELLO_DA_USARE.upper()}) ---")

    load_dotenv()
    
    print("Loading golden_dataset.json...")
    try:
        dataset_path = os.path.join(script_directory, 'golden_dataset.json')
        with open(dataset_path, 'r', encoding='utf-8') as f:
            golden_dataset = json.load(f)
        print(f"Dataset loaded successfully with {len(golden_dataset)} examples.")
    except FileNotFoundError:
        print(f"ERROR: File '{dataset_path}' not found.")
        sys.exit()

    print("Loading embedding model for evaluation scoring...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Embedding model loaded.")

    risultati_per_export = []
    risultati_valutazione_validi = []
    risultati_per_categoria_validi = {}

    print("\nStarting evaluation loop on the Golden Dataset...")
    for i, item in enumerate(golden_dataset):
        categoria = item['categoria']
        domanda_test = item["domanda"]
        risposta_ideale = item["risposta_ideale"]

        print(f"\n--- Evaluating item {i+1}/{len(golden_dataset)} (Category: {categoria}) ---")

        try:
            risposta_generata = esegui_rag(domanda_test)
            is_contesto_insufficiente = "contesto non sufficiente" in risposta_generata.lower()

            if is_contesto_insufficiente:
                print("   - 'Context not sufficient' response detected. Excluding from average calculation.")
                score = 0.0
            else:
                embedding_generato = model.encode([risposta_generata])
                embedding_ideale = model.encode([risposta_ideale])
                score = cosine_similarity(embedding_generato, embedding_ideale)[0][0]
            
            print("\n" + "="*20 + " RESPONSE COMPARISON " + "="*20)
            print(f"QUESTION        : {domanda_test}")
            print(f"IDEAL RESPONSE  :\n{risposta_ideale}\n")
            print(f"GENERATED RESPONSE:\n{risposta_generata.strip()}")
            print("="*62 + "\n")
            print(f"   - Semantic Similarity Score: {score:.4f}")
            
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}\n")
            score = 0.0
            risposta_generata = f"ERROR: {e}"
            is_contesto_insufficiente = True

        risultati_per_export.append({
            "categoria": categoria,
            "domanda": domanda_test,
            "risposta_ideale": risposta_ideale,
            "risposta_generata": risposta_generata.strip(),
            "score": score
        })
        
        if not is_contesto_insufficiente:
            risultati_valutazione_validi.append(score)
            if categoria not in risultati_per_categoria_validi:
                risultati_per_categoria_validi[categoria] = []
            risultati_per_categoria_validi[categoria].append(score)

    punteggio_medio = sum(risultati_valutazione_validi) / len(risultati_valutazione_validi) if risultati_valutazione_validi else 0
    
    punteggi_medi_categoria_dettagliati = {}
    for cat in set(item['categoria'] for item in golden_dataset):
        scores_validi = risultati_per_categoria_validi.get(cat, [])
        media = sum(scores_validi) / len(scores_validi) if scores_validi else 0
        totali_in_cat = sum(1 for item in golden_dataset if item['categoria'] == cat)
        punteggi_medi_categoria_dettagliati[cat] = {'media': media, 'validi': len(scores_validi), 'totali': totali_in_cat}

    print("\n\n--- FINAL EVALUATION RESULTS ---")
    print(f"Model Evaluated: {MODELLO_DA_USARE.upper()}")
    num_totali = len(golden_dataset)
    num_validi = len(risultati_valutazione_validi)
    print(f"Total Samples: {num_totali}")
    print(f"Evaluated Samples (excluding 'Context not sufficient'): {num_validi}")
    print(f"Average Score (on evaluated samples): {punteggio_medio:.4f}")
    
    print("\n--- DETAILED ANALYSIS BY CATEGORY ---")
    for categoria, dati in punteggi_medi_categoria_dettagliati.items():
        print(f"Category '{categoria}': Average Score = {dati['media']:.4f} ({dati['validi']}/{dati['totali']} evaluated samples)")

    salva_report_excel(risultati_per_export, punteggio_medio, punteggi_medi_categoria_dettagliati, num_validi, num_totali)

