# 📈 DataQuest: Real-Time Financial RAG

This model is a **real-time Retrieval-Augmented Generation (RAG)** engine built for financial analysis. It doesn't just "search" a static database—it listens to live stock markets, processes news streams, and reads uploaded documents instantly to provide up-to-the-second answers using **LLMs**.

---

## 🛠️ Tech Stack

This project leverages a high-performance stack designed for speed and real-time processing:

*   **⚡ Pathway**: The core engine for reactive data processing. It handles data ingestion, vector indexing, and RAG logic in real-time streaming mode.
*   **🧠 Groq (Llama 3)**: Ultra-low latency LLM inference. Used to generate human-like answers from the retrieved context.
*   **📊 Streamlit**: The interactive frontend dashboard for chatting and visualizing data.
*   **📈 Yahoo Finance & NewsAPI**: Live data sources for market quotes and global news.
*   **💾 SQLite**: Lightweight, robust storage for reliable data handoff between the high-speed backend and the UI.
*   **🐍 Python**: The primary language for all logic and integration.

---

## 🔄 How It Works (The Process)

The system operates in a continuous loop of **Ingestion**, **Processing**, and **Presentation**.

### 1. Ingestion Layer (The "Ears") 📡
The system continuously listens to three data sources:
*   **Market Data**: `app.py` runs a thread that fetches live stock prices (Yahoo Finance) and global news (NewsAPI) every minute. This data is written to a stream buffer (`stream_data.jsonl`).
*   **User Documents**: Any PDF or text file dropped into the `live_data/` folder (or uploaded via UI) is instantly detected, parsed, and injected into the stream.
*   **User Queries**: Questions typed in the dashboard are saved to `QnA/questions.csv`, which Pathway monitors as a stream.

### 2. Processing Core (The "Brain") 🧠
**Pathway** sits in the middle, treating all inputs as infinite streams:
1.  **Normalization**: It merges stock data, news, and uploaded files into a unified "Knowledge Stream".
2.  **Context Retrieval**: When a question arrives, Pathway dynamically filters the knowledge stream for relevant keywords and prioritizes recent data (Time-Decay Scoring).
3.  **LLM Inference**: The relevant context + the user's question are sent to **Groq**.
4.  **Answer Generation**: The LLM responds, and the result is written to an append-only log (`answers_log.csv`).

### 3. Sync & Presentation Layer (The "Face") 💻
To ensure a smooth user experience, we decouple the heavy processing from the UI:
*   **Sync Agent**: A dedicated thread watches `answers_log.csv` and safely updates a **SQLite Database** (`answers.db`). This prevents file locking issues or read errors.
*   **Dashboard**: The Streamlit app polls the SQLite DB. When your specific question's answer appears in the DB, it's displayed instantly in the chat.
*   **Live Ticker**: The dashboard also reads the processed news stream to show a running ticker of stock prices and breaking news on the sidebar.

---

## 🚀 Sequence of Events

Here is the step-by-step flow when you ask a question:

1.  **User** types: *"What's the latest on Apple?"* in the Dashboard.
2.  **Streamlit** saves the question to `questions.csv`.
3.  **Pathway** detects the new line in `questions.csv`.
4.  **Pathway** scans its real-time index for "Apple" (combining News + Stock Prices + Uploaded Docs).
5.  **Pathway** constructs a prompt with the top 5 most relevant pieces of information.
6.  **Groq API** receives the prompt and generates an answer.
7.  **Pathway** writes the answer to `answers_log.csv`.
8.  **Sync Thread** moves the answer to `answers.db`.
9.  **Streamlit** sees the answer in `answers.db` and updates the Chat UI.

**⚡ Total Latency:** Typically < 2 seconds.

---

## 🏃 Setup & Run

### 1. Prerequisites
*   Python 3.10+
*   API Keys:
    *   **Groq API Key** (for LLM)
    *   **Alpha Vantage** and **NewsAPI**

### 2. Installation
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (an example fiel has been added):
```ini
GROQ_API_KEY=gsk_...
ALPHA_VANTAGE_KEY=...
NEWSAPI_KEY=...
```

### 4. Running the App

**Terminal 1 (Backend):**
```bash
python app.py
```

**Terminal 2 (Frontend):**
```bash
python -m streamlit run dashboard.py
```
> Now you can open this using the localhost link ususally `http://localhost:8501/`
---

## 📂 Project Structure

*   `app.py`: The heart of the system. Runs Pathway streams, data fetchers, and the sync agent.
*   `dashboard.py`: The user interface built with Streamlit.
*   `entrypoint.sh`: Helper script to launch both services.
*   `live_data/`: Drop files here to "teach" the AI new knowledge instantly.
*   `QnA/`: Stores the conversation history and database.
