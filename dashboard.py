import streamlit as st
import pandas as pd
import time
import os
import datetime
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Real-Time Financial Analyst",
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

def load_live_answers(question_text=None):
    import sqlite3
    if not os.path.exists(ANSWERS_DB):
        return pd.DataFrame()
    try:
        with sqlite3.connect(ANSWERS_DB) as conn:
            # If specific question is requested, query for it directly
            if question_text:
                # Use parameter substitution for safety
                query = "SELECT timestamp, question, answer FROM answers WHERE question = ? ORDER BY timestamp DESC LIMIT 1"
                return pd.read_sql_query(query, conn, params=(question_text,))
            else:
                # Fallback to latest global answer
                return pd.read_sql_query("SELECT timestamp, question, answer FROM answers ORDER BY timestamp DESC LIMIT 1", conn)
    except Exception:
        return pd.DataFrame()

# --- Sidebar: Live Feed ---
with st.sidebar:
    st.markdown("### 📡 Live Intelligence")
   
    # New Chat Button (Simple)
    if st.button("➕ New Chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --- File Uploader (Drag & Drop) ---
    st.markdown("### 📂 Bytes & Data")
    uploaded_file = st.file_uploader("Drop PDF or TXT here", type=["txt", "pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        save_path = os.path.join("live_data", uploaded_file.name)
        # Handle duplicate filenames - append timestamp if needed, but for now overwrite or simple save
        # actually, let's just save it.
        
        if not os.path.exists("live_data"):
             os.makedirs("live_data")

        # Check file type
        if uploaded_file.type == "application/pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Save as txt
                txt_filename = os.path.splitext(uploaded_file.name)[0] + ".txt"
                save_path = os.path.join("live_data", txt_filename)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
                st.toast(f"✅ Converted & Injected: {txt_filename}", icon="🚀")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        else:
            # Assume TXT
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.toast(f"✅ Injected: {uploaded_file.name}", icon="🚀")

    st.divider()
    
    # Render News Feed
    @st.fragment(run_every=20)
    def render_news_feed():
        news_df = load_csv_data(NEWS_FILE)
        if not news_df.empty and 'timestamp' in news_df.columns:
            # Deduplicate by title to avoid repeated news
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            news_df = news_df.sort_values(by="timestamp", ascending=False).head(8)
            
            for _, row in news_df.iterrows():
                is_injection = row.get('source') == 'User Injection'
                css_class = "sidebar-news-item injected" if is_injection else "sidebar-news-item"
                icon = "📄" if is_injection else "🌐"
                title = row.get('title', 'Untitled')
                url = row.get('url', '#')
                
                # Handle potential NaN or empty values
                if pd.isna(url) or not str(url).strip() or str(url) == 'nan':
                    url = "#"
                    
                title_html = f'<a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">{title}</a>' if url != "#" else title
                
                # Add a hover effect or visual cue for links if needed, but for now keep it clean
                
                st.markdown(f"""
                <div class="{css_class}">
                    <div style="font-weight: 600; margin-bottom:2px;">{icon} {title_html}</div>
                    <div style="color: #888;">{row.get('content', '')[:60]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for data stream...")

    render_news_feed()

# --- Main Chat Area ---
st.title("📈 Real-Time Financial Analyst")

# --- Stock Ticker ---
@st.fragment(run_every=10)
def render_stock_ticker():
    import re
    
    df = load_csv_data(NEWS_FILE)
    if not df.empty and 'source' in df.columns:
        # Filter for Alpha Vantage data
        stock_df = df[df['source'] == 'Alpha Vantage'].copy()
        
        if not stock_df.empty:
            # Sort by timestamp descending
            stock_df = stock_df.sort_values(by="timestamp", ascending=False)
            
            # Deduplicate by symbol to keep only latest
            stock_df = stock_df.drop_duplicates(subset=['symbol'], keep='first')
            
            # Parse Content for Price and Change
            # Content format: "{Company} ({Symbol}) trading at ${price}, {change}% change."
            stocks = []
            for _, row in stock_df.iterrows():
                try:
                    content = row.get('content', '')
                    # Regex to extract price and change
                    # trading at $123.45, 1.23% change
                    price_match = re.search(r'trading at \$([\d\.]+)', content)
                    change_match = re.search(r',\s*([-\d\.]+)%\s*change', content)
                    
                    price = price_match.group(1) if price_match else "N/A"
                    change = change_match.group(1) if change_match else "0"
                    
                    # Determine coloring
                    try:
                        change_val = float(change)
                        trend = "🟢" if change_val >= 0 else "🔴"
                    except:
                        trend = "⚪"
                        
                    # Format timestamp
                    ts_raw = row.get('timestamp', '')
                    try:
                        # Handle potential fractional seconds or typical iso format
                        dt_obj = datetime.datetime.fromisoformat(str(ts_raw))
                        readable_ts = dt_obj.strftime("%b %d, %I:%M:%S %p")
                    except:
                        readable_ts = ts_raw

                    stocks.append({
                        "Symbol": row.get('symbol', 'N/A'),
                        "Price ($)": price,
                        "Change (%)": f"{trend} {change}%",
                        "Last Updated": readable_ts
                    })
                except:
                    continue
            
            if stocks:
                st.markdown("### 📊 Market Snapshot")
                final_df = pd.DataFrame(stocks)
                # Reorder columns
                final_df = final_df[["Symbol", "Price ($)", "Change (%)", "Last Updated"]]
                st.dataframe(final_df, width="stretch", hide_index=True)

render_stock_ticker()

st.divider()

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2. Chat Input (Gemini Style)
if prompt := st.chat_input("Ask about the market..."):
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
            # Query specifically for OUR question
            answers_df = load_live_answers(question_text=prompt)
            if not answers_df.empty:
                latest_row = answers_df.iloc[0]
                # Double check (redundant but safe)
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

