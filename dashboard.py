import streamlit as st
import pandas as pd
import time
import os
import datetime
import shutil
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Real-Time Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        h1 { color: #4CAF50; }
        .stButton>button { width: 100%; border-radius: 5px; }
        .stTextInput>div>div>input { border-radius: 5px; }
        .archive-banner { 
            background-color: #FFF3CD; 
            padding: 10px; 
            border-radius: 5px; 
            border-left: 4px solid #FFC107;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Real-Time Stock Analyst")
st.caption("Powered by Pathway | Live Data Injection Demo")

# Data Files
NEWS_FILE = "processed_news.csv"
ANSWERS_FILE = "./QnA/answers.csv" # Keep for backward compatibility/archiving if needed
ANSWERS_DB = "./QnA/answers.db"
QUESTIONS_FILE = "./QnA/questions.csv"
CHATS_DIR = "chats"

# Ensure chats directory exists
os.makedirs(CHATS_DIR, exist_ok=True)
if not os.path.exists("./QnA"):
    os.makedirs("./QnA", exist_ok=True)

# Initialize session state
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'live'
if 'selected_archive' not in st.session_state:
    st.session_state.selected_archive = None

def load_csv_data(file_path):
    """Load data from CSV file (for News and Archives)."""
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_live_answers():
    """Load latest answers directly from SQLite database."""
    import sqlite3
    
    if not os.path.exists(ANSWERS_DB):
        return pd.DataFrame(columns=["timestamp", "question", "answer", "context"])

    try:
        with sqlite3.connect(ANSWERS_DB) as conn:
            # Read all answers, sorted by most recent
            return pd.read_sql_query("SELECT timestamp, question, answer, context FROM answers ORDER BY timestamp DESC", conn)
    except Exception as e:
        # st.error(f"Error loading database: {e}")
        return pd.DataFrame()

def get_latest_state(df, key_cols):
    """
    Reconstructs the latest state of a table from a Pathway changelog (diff stream).
    """
    if df.empty:
        return df
        
    # Check if 'diff' and 'time' columns exist (Pathway outputs)
    if 'diff' not in df.columns or 'time' not in df.columns:
        return df

    # Sort by processing time (descending) to get latest updates first
    df_sorted = df.sort_values(by='time', ascending=False)
    
    # Deduplicate by key to get the most recent change for each item
    latest_updates = df_sorted.drop_duplicates(subset=key_cols)
    
    # Filter: Keep only rows where diff=1 (active/inserted). 
    # diff=-1 means the record was deleted/retracted in the most recent step.
    active_rows = latest_updates[latest_updates['diff'] == 1]
    
    return active_rows

def get_archived_chats():
    """Get list of archived chat files."""
    if not os.path.exists(CHATS_DIR):
        return []
    files = [f for f in os.listdir(CHATS_DIR) if f.endswith('.csv')]
    return sorted(files, reverse=True)

def archive_current_chat(chat_name=None):
    """Archive current chat with optional name."""
    # Check if there's any data to archive
    answers_df = load_live_answers()
    if answers_df.empty:
        return False, "No chat history to archive."
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if chat_name:
        # Sanitize chat name
        safe_name = "".join(c for c in chat_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"chat_{safe_name}_{timestamp}.csv"
    else:
        filename = f"chat_{timestamp}.csv"
    
    archive_path = os.path.join(CHATS_DIR, filename)
    
    # Copy answers file to archive
    if os.path.exists(ANSWERS_DB):
        # Dump DB to CSV for archiving
        df = load_live_answers()
        df.to_csv(archive_path, index=False)
    else:
        return False, "No database found."
    
    return True, f"Chat archived as: {filename}"

def clear_current_chat():
    """Clear current questions and answers."""
    # Clear questions file
    with open(QUESTIONS_FILE, 'w') as f:
        f.write("timestamp,query_text,user\n")
    
    # Clear answers file
    # For SQLite, we delete all rows
    import sqlite3
    if os.path.exists(ANSWERS_DB):
        with sqlite3.connect(ANSWERS_DB) as conn:
            conn.execute("DELETE FROM answers")
            conn.commit()
    
    return True

# Sidebar
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.subheader("Instructions")
st.sidebar.info(
    "1. **Status**: Check the Live Feed.\n"
    "2. **Ask**: Query about a topic.\n"
    "3. **Inject**: Drop a file into `live_data/`.\n"
    "4. **Ask Again**: See the answer change."
)

# Chat Session Management
st.sidebar.divider()
st.sidebar.subheader("💬 Chat Sessions")

# View mode selection
view_options = ["🟢 Live Chat"]
archived_chats = get_archived_chats()
if archived_chats:
    view_options.extend([f"📁 {chat}" for chat in archived_chats])

selected_view = st.sidebar.selectbox(
    "Select View",
    view_options,
    key="view_selector"
)

# Update session state based on selection
if selected_view.startswith("🟢"):
    st.session_state.view_mode = 'live'
    st.session_state.selected_archive = None
else:
    st.session_state.view_mode = 'archive'
    st.session_state.selected_archive = selected_view.replace("📁 ", "")

# Chat actions (only show in live mode)
if st.session_state.view_mode == 'live':
    col_clear, col_new = st.sidebar.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear", help="Clear current chat"):
            if st.session_state.get('confirm_clear', False):
                clear_current_chat()
                st.success("Chat cleared!")
                st.session_state.confirm_clear = False
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")
    
    with col_new:
        if st.button("➕ New Chat", help="Archive current chat and start new"):
            st.session_state.show_new_chat_dialog = True
    
    # New chat dialog
    if st.session_state.get('show_new_chat_dialog', False):
        with st.sidebar.form("new_chat_form"):
            st.write("**Start New Chat**")
            chat_name = st.text_input(
                "Chat name (optional)",
                placeholder="e.g., Market Research"
            )
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Create")
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if submit:
                success, message = archive_current_chat(chat_name if chat_name else None)
                if success:
                    clear_current_chat()
                    st.success(message)
                    st.success("New chat started!")
                else:
                    st.info(message)
                st.session_state.show_new_chat_dialog = False
                time.sleep(1)
                st.rerun()
            
            if cancel:
                st.session_state.show_new_chat_dialog = False
                st.rerun()

# Layout
col1, col2 = st.columns([1, 1])

# Show archive banner if viewing archive
if st.session_state.view_mode == 'archive':
    st.markdown(
        f'<div class="archive-banner">📁 <b>Viewing Archive:</b> {st.session_state.selected_archive}</div>',
        unsafe_allow_html=True
    )

# Q&A Section (only show in live mode)
if st.session_state.view_mode == 'live':
    with st.container():
        st.subheader("❓ Ask the Analyst")
        with st.form("query_form"):
            user_query = st.text_input("Enter your question:", placeholder="e.g. What is the latest update on Project X?")
            submitted = st.form_submit_button("Ask Question")
            if submitted and user_query:
                # Append to questions.csv
                new_q = pd.DataFrame([{
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query_text": user_query,
                    "user": "WebUser"
                }])
                # Append without header if exists
                header = not os.path.exists(QUESTIONS_FILE)
                new_q.to_csv(QUESTIONS_FILE, mode='a', header=header, index=False)
                st.success("Query sent to Analyst!")
                # time.sleep(1)

# Display Logic
placeholder_news = col1.empty()
placeholder_answers = col2.empty()

# Determine which data to display
if st.session_state.view_mode == 'archive' and st.session_state.selected_archive:
    answers_file = os.path.join(CHATS_DIR, st.session_state.selected_archive)
    auto_refresh = False
else:
    answers_file = "LIVE_DB" # Special flag for SQLite
    auto_refresh = st.checkbox("Auto-Refresh (2s)", value=True)

# Refresh Loop
if auto_refresh:
    while True:
        # News Feed
        with placeholder_news.container():
            st.subheader("🔴 Live Feed (Market & Injected)")
            news_df = load_csv_data(NEWS_FILE)
            if not news_df.empty and 'timestamp' in news_df.columns:
                news_df = news_df.sort_values(by="timestamp", ascending=False).head(5)
                for _, row in news_df.iterrows():
                    source_icon = "📄" if row.get('source') == 'User Injection' else "📡"
                    with st.expander(f"{source_icon} {row.get('title', 'Untitled')}"):
                        st.write(row.get('content', ''))
                        st.caption(f"{row.get('timestamp')}")
            else:
                st.info("Waiting for data...")

        # Answers
        with placeholder_answers.container():
            st.subheader("💡 Analyst Answers")
            
            if answers_file == "LIVE_DB":
                 # Load from SQLite
                 answers_df = load_live_answers()
                 # No need for get_latest_state, DB is already clean
            else:
                 # Load from CSV (Archive)
                 answers_df = load_csv_data(answers_file)
                 # Archives might interpret raw logs or clean snapshots? 
                 # Usually snapshots. So maybe no need for get_latest_state if archived from clean state.
                 # But let's keep it just in case archives were raw.
                 answers_df = get_latest_state(answers_df, key_cols=['timestamp', 'question'])

            if not answers_df.empty:
                # Ensure we have the columns
                if 'timestamp' in answers_df.columns:
                     answers_df = answers_df.sort_values(by="timestamp", ascending=False).head(10)
                
                for _, row in answers_df.iterrows():
                     q_text = row.get('question', 'Unknown')
                     a_text = row.get('answer', 'No answer')
                     st.success(f"**Q:** {q_text}\n\n**A:** {a_text}")
            else:
                st.info("No answers yet. Ask a question!")
        
        time.sleep(2)
else:
    # Static display (for archive view)
    with placeholder_news.container():
        st.subheader("🔴 Live Feed (Market & Injected)")
        news_df = load_csv_data(NEWS_FILE)
        if not news_df.empty and 'timestamp' in news_df.columns:
            news_df = news_df.sort_values(by="timestamp", ascending=False).head(5)
            for _, row in news_df.iterrows():
                source_icon = "📄" if row.get('source') == 'User Injection' else "📡"
                with st.expander(f"{source_icon} {row.get('title', 'Untitled')}"):
                    st.write(row.get('content', ''))
                    st.caption(f"{row.get('timestamp')}")
        else:
            st.info("Waiting for data...")

    with placeholder_answers.container():
        st.subheader("💡 Analyst Answers")
        # In static mode, we are usually viewing archives
        if answers_file == "LIVE_DB":
             answers_df = load_live_answers()
        else:
             answers_df = load_csv_data(answers_file)
             answers_df = get_latest_state(answers_df, key_cols=['timestamp', 'question'])

        if not answers_df.empty:
             if 'timestamp' in answers_df.columns:
                  answers_df = answers_df.sort_values(by="timestamp", ascending=False).head(10)
             
             for _, row in answers_df.iterrows():
                    q_text = row.get('question', 'Unknown')
                    a_text = row.get('answer', 'No answer')
                    st.success(f"**Q:** {q_text}\n\n**A:** {a_text}")
        else:
            st.info("No answers yet. Ask a question!")
