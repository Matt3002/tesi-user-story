# Progetto di Tesi: Ottimizzazione della Gestione delle User Story tramite Modelli RAG

Questo repository contiene il codice sorgente per il progetto di tesi di laurea "Ottimizzazione della Gestione delle User Story tramite Modelli RAG". Il progetto implementa e valuta un sistema di **Retrieval-Augmented Generation (RAG)** per assistere nella creazione di user story Agili contestualizzate e di alta qualità, confrontando diversi modelli linguistici e strategie di automazione.

---

## 🏛️ Architettura Finale

Il sistema utilizza un'architettura ibrida che combina servizi cloud per l'archiviazione e la ricerca con modelli linguistici eseguibili sia in cloud che in locale.



1.  **Data Ingestion (Azure Blob Storage):** I documenti sorgente (`.txt`, `.pdf`) che costituiscono la base di conoscenza sono archiviati in modo sicuro su Azure Blob Storage.
2.  **Embedding (Sentence Transformers):** Uno script locale utilizza il modello `paraphrase-multilingual-MiniLM-L12-v2` per trasformare i frammenti di testo (chunk) in vettori numerici (embeddings).
3.  **Indexing & Retrieval (Azure AI Search):** I chunk e i relativi vettori vengono indicizzati su Azure AI Search, che gestisce la ricerca semantica per recuperare il contesto più pertinente a una data domanda.
4.  **Generation (LLM - Azure OpenAI / Ollama):** Il contesto recuperato viene combinato con la domanda originale e un prompt strutturato, per poi essere inviato a un Large Language Model (come `gpt-4o` su Azure o `Llama 3.2` in locale) per la generazione della user story finale.

---

## 📂 Struttura del Progetto

* **`.github/workflows/`**: Contiene i file YAML per l'automazione con GitHub Actions.
    * `main.yml`: Workflow di valutazione automatica (CI) che si avvia ad ogni push.
    * `chatbot.yml`: Workflow per avviare manualmente il chatbot dall'interfaccia di GitHub.
* **`writer/`**: Moduli Python che gestiscono la comunicazione con i diversi LLM.
    * `writer_azure_openai.py`: Per il modello `gpt-4o` su Azure.
    * `writer_ollama.py`: Per i modelli locali tramite Ollama.
* **`01_crea_indice.py`**: Script per creare (una sola volta) lo schema dell'indice su Azure AI Search.
* **`02_popola_indice.py`**: Script per l'ingestione dei dati: legge i file da Blob Storage, crea gli embedding e popola l'indice.
* **`03_valuta_modello.py`**: Script per la valutazione quantitativa del sistema RAG completo.
* **`without_rag.py`**: Script per l'esperimento di controllo che valuta un LLM senza il recupero di contesto.
* **`chatbot.py`**: Un'interfaccia a riga di comando per dialogare in modo interattivo con l'agente RAG.
* **`index.py`**: Modulo che contiene la logica di **recupero (Retrieval)** da Azure AI Search.
* **`golden_dataset.json`**: Il dataset di test con 25+ esempi usati per la valutazione oggettiva del sistema.
* **`requirements.txt`**: Elenco di tutte le librerie Python necessarie.
* **`.env`**: File per la gestione delle credenziali e delle chiavi API (da non caricare su GitHub).

---

## ⚙️ Setup e Installazione

1.  **Clonare il Repository**
    ```bash
    git clone <URL_DEL_TUO_REPOSITORY>
    cd <NOME_CARTELLA_PROGETTO>
    ```

2.  **Configurare le Credenziali**
    * Crea un file chiamato `.env` nella cartella principale.
    * Copia il contenuto di `.env.example` (se presente) o inserisci le seguenti variabili con i tuoi valori:
        ```env
        AZURE_SEARCH_ENDPOINT="..."
        AZURE_SEARCH_API_KEY="..."
        AZURE_SEARCH_INDEX_NAME="..."
        AZURE_STORAGE_CONNECTION_STRING="..."
        AZURE_OPENAI_API_KEY="..."
        AZURE_OPENAI_ENDPOINT="..."
        ```

3.  **Installare le Dipendenze**
    Assicurati di avere Python 3.10+ installato. Dopodiché, crea un ambiente virtuale ed esegui:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Workflow Operativo

Per eseguire il progetto dall'inizio alla fine, segui questi passaggi nell'ordine:

1.  **Creare l'Indice su Azure** (da eseguire solo una volta):
    ```bash
    python 01_crea_indice.py
    ```

2.  **Popolare l'Indice con i Documenti**:
    ```bash
    python 02_popola_indice.py
    ```

3.  **Eseguire la Valutazione**:
    * Per valutare il sistema RAG completo (es. con Azure OpenAI):
        ```bash
        # Modifica la variabile MODELLO_DA_USARE in 03_valuta_modello.py
        python 03_valuta_modello.py
        ```
    * Per eseguire il test di controllo (Ollama senza RAG):
        ```bash
        python without_rag.py
        ```

4.  **Avviare il Chatbot Interattivo**:
    ```bash
    python chatbot.py
    ```

---

## 📊 Risultati della Valutazione

La valutazione ha messo a confronto diversi modelli, portando a scoperte significative sull'efficacia dell'architettura RAG.

| Configurazione | Punteggio Medio Globale | Campioni Valutati |
| :--- | :--- | :--- |
| **Azure OpenAI (gpt-4o) con RAG** | **0.8668** | 17 / 25 |
| **Ollama (Llama 3.2) con RAG** | 0.7532 | 25 / 25 |

**Conclusioni Chiave:**

* **Superiorità di `gpt-4o`:** Il modello di Azure OpenAI si è dimostrato nettamente superiore sia per la qualità semantica delle risposte sia per la sua **perfetta aderenza alle istruzioni di formattazione**.
* **Il Fallimento del Recupero è il Vero Limite:** La scoperta più importante è che `gpt-4o`, essendo molto "disciplinato", si è rifiutato di rispondere nel 32% dei casi, segnalando "Contesto non sufficiente". Questo ha dimostrato che il vero collo di bottiglia del sistema non è la generazione, ma la **qualità del recupero (Retrieval)**.
* **I Modelli Locali Sono Competitivi ma Indisciplinati:** `Llama 3.2` ha mostrato una comprensione semantica molto buona ma una scarsa capacità di seguire le regole di formattazione, rendendo il suo output meno affidabile per un'automazione completa.

---

## 🤖 Automazione con GitHub Actions

Il progetto include due workflow di CI/CD per l'automazione:
* **Valutazione Continua:** Ad ogni `push` sul branch `main`, il workflow `main.yml` esegue automaticamente la valutazione completa con Azure OpenAI per monitorare la qualità.
* **Chatbot su Richiesta:** Il workflow `chatbot.yml` permette di eseguire il chatbot direttamente dall'interfaccia di GitHub per test e dimostrazioni.

Per abilitarli, è necessario configurare le proprie credenziali come **Secrets** nel repository GitHub (`Settings > Secrets and variables > Actions`).
