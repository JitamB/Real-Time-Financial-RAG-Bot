# 📈 DataQuest: Real-Time Financial RAG Dashboard

DataQuest is a real-time Retrieval-Augmented Generation (RAG) application designed for financial analysts. It continuously monitors live stock market data and news feeds, enabling users to ask questions and receive up-to-the-second answers based on the absolute latest information.

Powered by **Pathway** for reactive data processing, **Groq (Llama 3)** for ultra-fast inference, and **Streamlit** for a responsive user interface.

---

## 🏗️ Architecture

The application caches no stale data; it processes streams reactively.

### 1. The Input Layer (Real-Time Ingestion)
*   **Live Data Stream (`app.py`):** Fetches real-time stock quotes (Alpha Vantage) and news articles (NewsAPI) every 60 seconds.
*   **File Injection:** Monitors `live_data/` for manually dropped PDF reports or text files.
*   **User Questions:** Reads incoming questions from the Dashboard via `questions.csv`.

### 2. The Processing Core (Pathway RAG)
*   **Vector Search & Context:** Pathway indexes incoming data streams on-the-fly.
*   **LLM Inference:** Relevant context is sent to **Groq (Llama 3.3-70b)** to generate an answer.
*   **Logging:** Raw results are written to an append-only log: `answers_log.csv`.

### 3. The Data Sync Layer (Robust Persistence)
*   **Background Syncer:** A dedicated thread in `app.py` tails `answers_log.csv`.
*   **Cleaning & Upsert:** It uses robust Pandas parsing to handle complex text, filters invalid entries, and updates a **SQLite Database (`answers.db`)**.
*   **Concurrency:** This decouples the high-speed RAG pipeline from the user-facing dashboard, preventing file locks and data corruption.

### 4. The Presentation Layer (Streamlit)
*   **Dashboard (`dashboard.py`):** Connects to `answers.db` via SQL.
*   **Features:** Displays chat history, supports "Clear Chat" (truncates DB), and "Archive Chat" (dumps to CSV).

---

## 🚀 Setup & Run Instructions

### 1. Prerequisites
*   Python 3.10+
*   API Keys for: **Groq**, **Alpha Vantage**, and **NewsAPI**.

### 2. Environment Setup
1.  **Clone/Download** the repository.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```ini
    GROQ_API_KEY=gsk_...
    ALPHA_VANTAGE_KEY=...
    NEWSAPI_KEY=...
    ```

### 3. Running the Application
You need to run the Backend (Pathway) and Frontend (Streamlit) in separate terminals.

**Terminal 1: Start the Backend (Pathway)**
This runs the continuous data processing pipeline.
```bash
python app.py
```
*Wait until you see "Pathway is running..."*

**Terminal 2: Start the Frontend (Dashboard)**
This launches the web interface.
```bash
python -m streamlit run dashboard.py
```

### 4. Usage
*   Open the URL shown in Terminal 2 (usually `http://localhost:8501`).
*   Type a question (e.g., *"What is the latest on AAPL?"*).
*   The system will fetch the latest context and generate an answer.