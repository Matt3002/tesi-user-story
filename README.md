# Progetto di Tesi: Gestione di User Story con Modello RAG

Questo repository contiene il codice sorgente per il progetto di tesi di laurea "Ottimizzazione della Gestione delle User Story tramite Modelli RAG". Il progetto implementa un sistema di Retrieval-Augmented Generation (RAG) per assistere nella creazione di user story Agile contestualizzate e di alta qualità.

## 🏛️ Architettura

Il sistema utilizza un'architettura basata su servizi cloud per orchestrare il flusso dei dati:

* **Azure Blob Storage**: Utilizzato come repository per i documenti sorgente (PDF, TXT) che costituiscono la base di conoscenza.
* **Azure AI Search**: Funge da motore di indicizzazione e ricerca vettoriale per recuperare il contesto pertinente.
* **Google Gemini API**: Utilizzata sia per generare gli embeddings (vettori semantici) dei documenti, sia per la generazione finale del testo (user story).



## 📂 Struttura del Progetto

* `1_crea_indice.py`: Script per creare l'indice su Azure AI Search tramite chiamata API diretta.
* `2_popola_indice.py`: Script per estrarre il testo dai documenti su Blob Storage, creare embeddings con Gemini e popolare l'indice.
* `3_valuta_modello.py`: Script per eseguire una valutazione quantitativa del modello RAG utilizzando un golden dataset.
* `index.py`: Modulo che contiene la logica per la ricerca e il recupero del contesto da Azure AI Search.
* `writer/writer_gemini.py`: Modulo che contiene la logica per la generazione della user story con Gemini.
* `test_singola_domanda.py`: Uno script di utility per testare il sistema con una singola domanda.
* `golden_dataset.json`: Il dataset di test con 20 esempi per la valutazione oggettiva del sistema.

## ⚙️ Setup e Installazione

1.  **Clonare il Repository**
    ```bash
    git clone <URL_DEL_TUO_REPOSITORY>
    cd <NOME_CARTELLA>
    ```

2.  **Configurare le Credenziali**
    * Rinomina il file `.env.example` (che dovrai creare) in `.env`.
    * Inserisci le tue chiavi API e gli endpoint per i servizi Azure e Google Gemini.

3.  **Installare le Dipendenze**
    Assicurati di avere Python 3.10+ installato. Dopodiché, esegui:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Come Usare il Progetto (Workflow)

Per eseguire il progetto dall'inizio alla fine, segui questi passaggi nell'ordine:

1.  **Creare l'Indice su Azure**
    Questo passaggio va eseguito solo una volta.
    ```bash
    python 1_crea_indice.py
    ```

2.  **Popolare l'Indice con i Tuoi Documenti**
    Assicurati che i tuoi file `.pdf` e `.txt` siano nel container corretto su Azure Blob Storage.
    ```bash
    python 2_popola_indice.py
    ```

3.  **Eseguire la Valutazione Automatica**
    Per misurare le performance del sistema, lancia lo script di valutazione.
    ```bash
    python 3_valuta_modello.py
    ```

## 📊 Risultati della Valutazione

L'esecuzione dello script di valutazione sul `golden_dataset.json` ha prodotto un **punteggio medio di similarità semantica di 0.6905**. L'analisi dettagliata per categorie ha dimostrato l'efficacia del sistema con richieste chiare e complesse (media > 0.72) e la sua robustezza con richieste fuori contesto, validando l'approccio RAG.