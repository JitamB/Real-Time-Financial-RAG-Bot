import streamlit as st
import pandas as pd
import time
import os
import datetime
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Real-Time Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("styles.css"):
    load_css("styles.css")

# --- Constants & Setup ---
NEWS_FILE = "processed_news.csv"
ANSWERS_DB = "./QnA/answers.db"
QUESTIONS_FILE = "./QnA/questions.csv"
CHATS_DIR = "chats"

os.makedirs(CHATS_DIR, exist_ok=True)
if not os.path.exists("./QnA"):
    os.makedirs("./QnA", exist_ok=True)

# Initialize session state for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Data Logic ---
def load_csv_data(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, on_bad_lines='skip')
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def load_live_answers():
    import sqlite3
    if not os.path.exists(ANSWERS_DB):
        return pd.DataFrame()
    try:
        with sqlite3.connect(ANSWERS_DB) as conn:
            # Get latest answer
            return pd.read_sql_query("SELECT timestamp, question, answer FROM answers ORDER BY timestamp DESC LIMIT 1", conn)
    except Exception:
        return pd.DataFrame()

# --- Sidebar: Live Feed ---
with st.sidebar:
    st.markdown("### 📡 Live Intelligence")
   
    # New Chat Button (Simple)
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    # Render News Feed
    news_df = load_csv_data(NEWS_FILE)
    if not news_df.empty and 'timestamp' in news_df.columns:
        news_df = news_df.sort_values(by="timestamp", ascending=False).head(8)
        
        for _, row in news_df.iterrows():
            is_injection = row.get('source') == 'User Injection'
            css_class = "sidebar-news-item injected" if is_injection else "sidebar-news-item"
            icon = "📄" if is_injection else "🌐"
            title = row.get('title', 'Untitled')
            # time_str = row.get('timestamp', '')[11:16] # Extract HH:MM
            
            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-weight: 600; margin-bottom:2px;">{icon} {title}</div>
                <div style="color: #888;">{row.get('content', '')[:60]}...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Waiting for data stream...")

# --- Main Chat Area ---
st.title("📈 Real-Time Stock Analyst")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2. Chat Input (Gemini Style)
if prompt := st.chat_input("Ask about the market or injected files..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process Query
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analyzing market data...")
        
        # 1. Write prompt to CSV for Backend
        new_q = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "query_text": prompt,
            "user": "WebUser"
        }])
        header = not os.path.exists(QUESTIONS_FILE)
        new_q.to_csv(QUESTIONS_FILE, mode='a', header=header, index=False)
        
        # 2. Poll for Answer (Simple Polling)
        # We wait up to 15 seconds for a new answer to appear in DB
        # This is a synchronous block that blocks the UI, which is fine for direct response
        time_to_wait = 45
        found_answer = False
        
        # Get start len
        start_time = datetime.datetime.now()
        
        import sqlite3
        
        # Simple Logic: Check if recent answer matches our query text
        # Or just wait for ANY new answer with timestamp > now
        # Ideally, we'd add an ID, but for this demo, keyword matching or timestamp is okay.
        
        while (datetime.datetime.now() - start_time).seconds < time_to_wait:
            time.sleep(1)
            answers_df = load_live_answers()
            if not answers_df.empty:
                latest_row = answers_df.iloc[0]
                # Check if this answer corresponds to our current prompt (by matching question text)
                # This is safer than timestamp comparison which can be tricky with timezone/precision
                db_question = latest_row.get('question', '')
                if db_question == prompt:
                     # It's a match!
                     final_answer = latest_row['answer']
                     message_placeholder.markdown(final_answer)
                     # Add to history
                     st.session_state.messages.append({"role": "assistant", "content": final_answer})
                     found_answer = True
                     break
        
        if not found_answer:
            timeout_msg = "⏱️ Analysis timed out. The backend might be busy."
            message_placeholder.markdown(timeout_msg)
            st.session_state.messages.append({"role": "assistant", "content": timeout_msg})

