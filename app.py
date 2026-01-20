import pathway as pw
import os
import time
import json
import threading
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
DATA_FILE = "stream_data.jsonl"
# DATA_FILE = "processed_news.csv"
QUESTIONS_FILE = "./QnA/questions.csv"
ANSWERS_DB = "./QnA/answers.db"
ANSWERS_LOG_FILE = "./QnA/answers_log.csv" # Internal changelog
LIVE_DATA_DIR = "live_data/"
FILE_TRACKER = "file_tracker.json"  # Track processed files to avoid duplicates

# --- Real Data Fetcher (Alpha Vantage + NewsAPI) ---
def real_data_stream():
    """Fetch real market data and news using APIs."""
    import requests
    
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY")
    # Map symbols to company names for better RAG context matching
    stock_map = {
        "AAPL": "Apple",
        "GOOGL": "Google Alphabet",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "META": "Meta Facebook",
        "NFLX": "Netflix",
        "AMD": "AMD",
        "INTC": "Intel"
    }
    
    print(f"Starting REAL data stream to {DATA_FILE}...")
    print(f"Alpha Vantage Key: {'✓ Set' if alpha_vantage_key else '✗ Missing'}")
    print(f"NewsAPI Key: {'✓ Set' if newsapi_key else '✗ Missing'}")
    
    while True:
        try:
            # Fetch stock data from Yahoo Finance (yfinance)
            import yfinance as yf
            
            for symbol, company_name in stock_map.items():
                try:
                    ticker = yf.Ticker(symbol)
                    # fast_info is usually faster than .info
                    info = ticker.fast_info
                    current_price = info.last_price
                    prev_close = info.previous_close
                    
                    if current_price and prev_close:
                        change_p = ((current_price - prev_close) / prev_close) * 100
                        change_percent = f"{change_p:.2f}"
                        price = f"{current_price:.2f}"
                        
                        stock_data = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "symbol": symbol,
                            "title": f"Real-time update for {company_name} ({symbol})",
                            "content": f"{company_name} ({symbol}) trading at ${price}, {change_percent}% change.",
                            "source": "Yahoo finance",
                            "url": f"https://finance.yahoo.com/quote/{symbol}"
                        }
                        
                        with open(DATA_FILE, "a") as f:
                            f.write(json.dumps(stock_data) + "\n")
                            f.flush()
                        print(f"✓ Fetched {symbol}: ${price} ({change_percent}%)")
                    else:
                         print(f"Skipping {symbol}: No price data")
                         
                except Exception as e:
                    print(f"Error fetching {symbol} from yfinance: {e}")
                
            time.sleep(20)

            
            # Fetch news from NewsAPI
            if newsapi_key and newsapi_key != "your-newsapi-key":
                try:
                    # Dynamically build query
                    symbols_query = " OR ".join(stock_map.keys())
                    url = f"https://newsapi.org/v2/everything?q=stock+market+OR+{symbols_query}&language=en&sortBy=publishedAt&pageSize=5&apikey={newsapi_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        news_data = response.json()
                        if "articles" in news_data:
                            for article in news_data["articles"][:3]:  # Get top 3 articles
                                news_item = {
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "symbol": "NEWS",
                                    "title": article.get("title", "No title"),
                                    "content": article.get("description", "") or article.get("content", ""),
                                    "source": article.get("source", {}).get("name", "NewsAPI"),
                                    "url": article.get("url", "#")
                                }
                                
                                with open(DATA_FILE, "a") as f:
                                    f.write(json.dumps(news_item) + "\n")
                                    f.flush()
                            print(f"✓ Fetched {len(news_data['articles'][:3])} news articles")
                except Exception as e:
                    print(f"Error fetching news from NewsAPI: {e}")
            
            # Wait before next iteration (fetch every 5 minutes)
            print(f"Waiting 5 minutes before next fetch...")
            time.sleep(300)
            
        except Exception as e:
            print(f"Error in data stream: {e}")
            time.sleep(60)

# --- Snapshot Syncer (Maintains single answer per question) ---
def sync_snapshot_file():
    """
    Reads the log file using Pandas (robust for multi-line CSVs) and updates SQLite.
    We revert to full-file reading for correctness, as manual tailing is brittle with CSV quotes.
    """
    import sqlite3
    import pandas as pd
    import time
    
    print("Starting snapshot syncer (Pandas -> SQLite)...")
    
    init_db()
    
    # 0. Clean DB on Start (Remove garbage from previous parser errors)
    try:
        with sqlite3.connect(ANSWERS_DB) as conn:
            conn.execute("DELETE FROM answers WHERE timestamp NOT LIKE '20%'")
            conn.commit()
    except Exception:
        pass
    
    while True:
        try:
            if os.path.exists(ANSWERS_LOG_FILE):
                # Read the log securely (handles multiline quotes automatically)
                try:
                    # check if file is empty
                    if os.path.getsize(ANSWERS_LOG_FILE) == 0:
                         time.sleep(1)
                         continue
                         
                    # Use Python engine with backslash escaping (Pathway seems to use backslash for quotes)
                    df = pd.read_csv(ANSWERS_LOG_FILE, engine='python', on_bad_lines='skip')
                except Exception:
                    # File might be locked or writing
                    time.sleep(1)
                    continue
                
                if not df.empty and 'timestamp' in df.columns and 'question' in df.columns:
                     # DEBUG LOGGING
                     with open("debug_sync.txt", "a") as f:
                         f.write(f"{datetime.datetime.now()} - Read {len(df)} rows. Columns: {list(df.columns)}\n")

                     # Filter for relevant columns
                     # We only care about the latest answer for each question
                     
                     # 1. Sort by time/timestamp to get latest
                     # If 'time' column exists (Pathway timestamp), use it. Else use 'timestamp'
                     sort_col = 'time' if 'time' in df.columns else 'timestamp'
                     df_sorted = df.sort_values(by=sort_col, ascending=True)
                     
                     with sqlite3.connect(ANSWERS_DB) as conn:
                         cursor = conn.cursor()
                         
                         for _, row in df_sorted.iterrows():
                             # Extract fields
                             ts = str(row.get('timestamp', ''))
                             q = str(row.get('question', ''))
                             ans = str(row.get('answer', ''))
                             ctx = str(row.get('context_used', '') or row.get('context', ''))
                             
                             # Strict Filter:
                             # 1. Skip if question is empty or timestamp doesn't look like a year (202...)
                             if not q or not str(ts).startswith('202'):
                                 with open("debug_sync.txt", "a") as f:
                                     f.write(f"Skipping row: q={q}, ts={ts}\n")
                                 continue
                            
                             # 2. Skip if question looks like a diff integer (1 or -1) due to misalignment
                             if q in ['1', '-1', '1.0', '-1.0']:
                                 continue

                             # 3. Ensure we only process 'active' rows if 'diff' exists
                             # If diff is in columns, we only want diff=1.
                             # If diff is mistakenly read as question, we caught it in step 2.
                             if 'diff' in row:
                                 try:
                                     d = float(row['diff'])
                                     if d != 1:
                                         continue
                                 except:
                                     pass

                             cursor.execute("""
                                 INSERT OR REPLACE INTO answers (timestamp, question, answer, context)
                                 VALUES (?, ?, ?, ?)
                             """, (ts, q, ans, ctx))
                             
                             if os.path.exists("debug_sync.txt") and os.path.getsize("debug_sync.txt") < 50000:
                                 with open("debug_sync.txt", "a") as f:
                                     f.write(f"Inserted: {q} at {ts}\n")
                         
                         conn.commit()
            
            time.sleep(1) # Poll every second
            
        except Exception as e:
            print(f"Snapshot sync error: {e}") 
            time.sleep(1)

# --- Pruning Mechanism (Prevents indefinite log growth) ---
def run_maintenance_tasks():
    """
    Periodically prunes:
    1. QUESTIONS_FILE (keep last 100 mins) - keeps Pathway graph small.
    2. ANSWERS_LOG_FILE (keep last 10 days) - prevents disk swell.
    3. ANSWERS_DB (keep last 10 days) - prevents DB bloat.
    """
    import pandas as pd
    import time
    import datetime
    import sqlite3
    
    print("Starting maintenance/pruning task...")
    
    while True:
        try:
            now = datetime.datetime.now()
            
            # 1. Prune QUESTIONS_FILE (100 mins)
            if os.path.exists(QUESTIONS_FILE) and os.path.getsize(QUESTIONS_FILE) > 0:
                try:
                    df = pd.read_csv(QUESTIONS_FILE)
                    if 'timestamp' in df.columns:
                        df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        cutoff_q = now - datetime.timedelta(minutes=100)
                        df_new = df[df['dt'] > cutoff_q].drop(columns=['dt'])
                        if len(df_new) < len(df):
                            df_new.to_csv(QUESTIONS_FILE, index=False)
                            print(f"✓ Pruned {len(df) - len(df_new)} old questions.")
                except Exception as e:
                     print(f"Pruning questions error: {e}")

            # 2. Prune ANSWERS_DB (10 Days)
            try:
                if os.path.exists(ANSWERS_DB):
                    cutoff_db = (now - datetime.timedelta(days=10)).isoformat()
                    with sqlite3.connect(ANSWERS_DB) as conn:
                        conn.execute("DELETE FROM answers WHERE timestamp < ?", (cutoff_db,))
                        conn.commit()
            except Exception as e:
                print(f"Pruning DB error: {e}")

            # 3. Prune ANSWERS_LOG_FILE (10 Days)
            # Note: This is a CSV used by Pathway. Pruning it might race with writing,
            # but since it's an append-only log, a quick rewrite is usually fine in this low-volume setting.
            if os.path.exists(ANSWERS_LOG_FILE) and os.path.getsize(ANSWERS_LOG_FILE) > 0:
                try:
                    # Only prune if file is getting somewhat large (>5MB) to avoid constant rewriting
                    if os.path.getsize(ANSWERS_LOG_FILE) > 5 * 1024 * 1024:
                        df = pd.read_csv(ANSWERS_LOG_FILE, on_bad_lines='skip', engine='python')
                        # Check for 'timestamp' or 'time'
                        ts_col = 'time' if 'time' in df.columns else 'timestamp'
                        if ts_col in df.columns:
                            df['dt'] = pd.to_datetime(df[ts_col], errors='coerce')
                            cutoff_log = now - datetime.timedelta(days=10)
                            df_new = df[df['dt'] > cutoff_log].drop(columns=['dt'])
                            
                            if len(df_new) < len(df):
                                df_new.to_csv(ANSWERS_LOG_FILE, index=False)
                                print(f"✓ Pruned {len(df) - len(df_new)} old log entries from answers_log.")
                except Exception as e:
                     print(f"Pruning log error: {e}")
            
            # Run every 100 minutes (matches the prune interval roughly)
            time.sleep(6000)
            
        except Exception as e:
            print(f"Pruning loop error: {e}")
            time.sleep(6000)

# --- Caching Mechanism ---
from collections import OrderedDict

class SimpleLRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
    def __len__(self):
        return len(self.cache)

ANSWER_CACHE = SimpleLRUCache(1000)

def init_db():
    import sqlite3
    with sqlite3.connect(ANSWERS_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                timestamp TEXT,
                question TEXT,
                answer TEXT,
                context TEXT,
                PRIMARY KEY (timestamp, question)
            )
        """)
        conn.commit()

def load_cache():
    """Load existing answers from SQLite to RAM."""
    import sqlite3
    global ANSWER_CACHE
    
    # Ensure DB exists
    init_db()
    
    if os.path.exists(ANSWERS_DB):
        try:
            with sqlite3.connect(ANSWERS_DB) as conn:
                cursor = conn.cursor()
                # Load latest 1000
                cursor.execute("SELECT timestamp, question, answer FROM answers ORDER BY timestamp DESC LIMIT 1000")
                rows = cursor.fetchall()
                for ts, q, ans in rows:
                    key = (str(ts), str(q))
                    ANSWER_CACHE.put(key, ans)
            print(f"✓ Loaded {len(ANSWER_CACHE)} answers from SQLite cache.")
        except Exception as e:
            print(f"Warning: Could not load answer cache: {e}")

# Helper for literals since pw.lit might be missing
@pw.udf
def literal(val, _ignored=None):
    return val

# --- Pathway Pipeline ---
def run_pipeline():
    # Load cache before starting
    load_cache()

    # 1. Stream 1: Background Market Data (JSONL)
    class NewsSchema(pw.Schema):
        timestamp: str
        symbol: str
        title: str
        content: str
        source: str
        url: str

    background_stream = pw.io.fs.read(
        DATA_FILE,
        format="json",
        schema=NewsSchema,
        mode="streaming",
        with_metadata=False
    )

    # Note: path must be to the directory
    # Use binary to read the whole file as a single blob, then decode
    live_files_stream = pw.io.fs.read(
        LIVE_DATA_DIR,
        format="binary",
        mode="streaming",
        with_metadata=False
    )
    
    # Track processed files to avoid duplicates
    import hashlib
    processed_files = {}
    
    # Load existing file tracker
    if os.path.exists(FILE_TRACKER):
        try:
            with open(FILE_TRACKER, 'r') as f:
                processed_files = json.load(f)
        except:
            processed_files = {}
    
    @pw.udf
    def get_file_hash_and_timestamp(content_bytes):
        """Generate hash from content and return submission timestamp."""
        import hashlib
        import datetime
        # content is bytes now
        if isinstance(content_bytes, str):
            content_bytes = content_bytes.encode('utf-8')
            
        file_hash = hashlib.md5(content_bytes).hexdigest()
        
        # Check if file was already processed
        if file_hash in processed_files:
            # Return existing timestamp to maintain consistency
            return processed_files[file_hash]
        else:
            # New file - record current timestamp
            timestamp = datetime.datetime.now().isoformat()
            processed_files[file_hash] = timestamp
            # Save to tracker file
            try:
                with open(FILE_TRACKER, 'w') as f:
                    json.dump(processed_files, f)
            except:
                pass
            return timestamp
    
    @pw.udf
    def decode_and_extract_info(content_bytes):
        """Decode bytes to string and extract title."""
        try:
            text = content_bytes.decode('utf-8')
        except:
             # Fallback to latin-1 or ignore errors
            text = content_bytes.decode('utf-8', errors='ignore')
            
        # Use first 50 chars as title
        first_line = text.split('\n')[0][:50]
        title = first_line if first_line else "Injected Data"
        return title, text

    # Select and unpack title/text
    files_with_text = live_files_stream.select(
        timestamp=get_file_hash_and_timestamp(pw.this.data),
        decoded_info=decode_and_extract_info(pw.this.data)
    )

    files_text = files_with_text.select(
        timestamp=pw.this.timestamp,
        symbol=literal("INJECTED", pw.this.timestamp),
        title=pw.this.decoded_info[0],
        content=pw.this.decoded_info[1],
        source=literal("User Injection", pw.this.timestamp),
        url=literal("#", pw.this.timestamp)
    )

    # Merge all data streams - treat equally
    all_news = pw.Table.concat(
        background_stream.promise_universes_are_disjoint(files_text),
        files_text
    )

    # 3. Questions Stream
    class QuerySchema(pw.Schema):
        timestamp: str
        query_text: str
        user: str

    query_stream = pw.io.fs.read(
        QUESTIONS_FILE,
        format="csv",
        schema=QuerySchema,
        mode="streaming",
        with_metadata=False
    )

    # 4. RAG / Analysis Logic
    
    @pw.udf
    def format_news(title, content, timestamp):
        return f"{title}: {content} ({timestamp})"

    news_with_key = all_news.select(
        key=literal(1, pw.this.timestamp), 
        content=format_news(pw.this.title, pw.this.content, pw.this.timestamp)
    )
    
    combined_news = news_with_key.groupby(pw.this.key).reduce(
        key=pw.this.key,
        context=pw.reducers.tuple(pw.this.content) 
    )
    
    @pw.udf
    def format_context(news_tuple, query):
        """Select top 5 most relevant recent articles based on query and timestamp."""
        if not news_tuple:
            return ""
        
        import re
        from datetime import datetime
        import string
        
        # Parse and score each article
        articles = []
        
        # Stop words for better keyword extraction
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 
                     'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 
                     'what', 'who', 'when', 'where', 'why', 'how', 'this', 'that'}
        
        # Extract keywords from query
        # Replace punctuation with spaces to handle "Jitam's" -> "Jitam s"
        query_clean = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        query_words = [
            w.lower().strip() 
            for w in query_clean.split() 
            if len(w) >= 2 and w.lower().strip() not in stop_words
        ]
        
        for item in news_tuple:
            # Parse timestamp from item (format: "title: content (timestamp)")
            timestamp_match = re.search(r'\(([^)]+)\)$', item)
            timestamp_str = timestamp_match.group(1) if timestamp_match else None
            
            # Calculate relevance score based on keyword matching
            item_lower = item.lower()
            match_count = sum(1 for word in query_words if word and word in item_lower)
            
            # Parse timestamp for recency scoring
            try:
                if timestamp_str:
                    ts = datetime.fromisoformat(timestamp_str)
                    # Recency score: more recent = higher score (hours ago)
                    hours_ago = (datetime.now() - ts).total_seconds() / 3600
                    recency_score = max(0, 100 - hours_ago)  # Decay over time
                else:
                    recency_score = 0
            except:
                recency_score = 0
            
            # Combined score: keyword relevance + recency
            # Weight: Match count * 50 means even 1 keyword match (50) beats perfect recency (30)
            combined_score = (match_count * 50.0) + (recency_score * 0.3)
            
            articles.append({
                'content': item,
                'score': combined_score,
                'match_count': match_count,
                'timestamp': timestamp_str
            })
        
        # Sort by combined score (relevance + recency)
        articles.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 5 most relevant articles
        top_articles = articles[:7]
        
        # If less than 5 articles match, fill with most recent articles
        # if len(top_articles) < 5:
        #     # Get articles not in top list, sorted by recency
        #     remaining = [a for a in articles if a not in top_articles]
        #     remaining.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        #     top_articles.extend(remaining[:5-len(top_articles)])
        
        # Return formatted context
        return "\n---\n".join([a['content'] for a in top_articles])

    # We need to pass query to format_context, so we'll do this after the join
    # For now, create a placeholder context table
    latest_context_table = combined_news.select(
        key=pw.this.key,
        all_context=pw.this.context  # Keep all context for now
    )

    # Prepare Query
    query_with_key = query_stream.select(
        key=literal(1, pw.this.timestamp),
        original_query=pw.this.query_text,
        user=pw.this.user,
        q_time=pw.this.timestamp
    )

    # Join Query with Context
    joined = query_with_key.join(latest_context_table, pw.left.key == pw.right.key)
    
    # Now format context based on query
    joined_with_context = joined.select(
        q_time=pw.this.q_time,
        original_query=pw.this.original_query,
        user=pw.this.user,
        context_str=format_context(pw.this.all_context, pw.this.original_query)
    )

    # Answer Generation
    @pw.udf
    def answer_with_llm_cache(query, context, timestamp):  # Renamed to force recompilation
        if not query:
            return "Empty question."
        
        # Check Cache
        key = (str(timestamp), str(query))
        cached_ans = ANSWER_CACHE.get(key)
        if cached_ans:
            return cached_ans
            
        if not context:
            return "No relevant data found yet."
        
        # Real LLM Call with Groq
        # Check env for key.
        groq_api_key = os.getenv("GROQ_API_KEY") 
        if groq_api_key and groq_api_key.startswith("gsk_"):
            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a helpful financial analyst. Answer based on the provided context. Ignore any \\n present in the context, don't write them in the answer"},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                answer = response.choices[0].message.content
                # Update Cache
                ANSWER_CACHE.put(key, answer)
                return answer
            except Exception as e:
                return f"Groq LLM Error: {e}"
        else:
            # Mock Logic - return summary of top articles
            context_lines = context.split("\n---\n")
            
            if context_lines:
                summary = f"Based on {len(context_lines)} relevant articles:\n\n"
                for i, line in enumerate(context_lines[:3], 1):
                    # Extract key info from each article
                    if ':' in line:
                        title_part = line.split(':')[0]
                        summary += f"{i}. {title_part}\n"
                
                ANSWER_CACHE.put(key, summary)
                return summary
            
            return "No relevant information found."

    results = joined_with_context.select(
        timestamp=pw.this.q_time,
        question=pw.this.original_query,
        answer=answer_with_llm_cache(pw.this.original_query, pw.this.context_str, pw.this.q_time),
        context_used=pw.this.context_str
    )
    
    # Output
    # Write stream to LOG file, not the user-facing file
    pw.io.csv.write(results, ANSWERS_LOG_FILE)
    
    # Also write processing news for debug/dashboard
    pw.io.csv.write(all_news, "processed_news.csv")
    
    pw.run()

if __name__ == "__main__":
    # Ensure structure
    if not os.path.exists("./QnA"):
        os.makedirs("./QnA")
    if not os.path.exists(LIVE_DATA_DIR):
        os.makedirs(LIVE_DATA_DIR)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f: pass
    if not os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "w") as f:
            f.write("timestamp,query_text,user\n")

    # Start Background Thread - REAL DATA
    t = threading.Thread(target=real_data_stream, daemon=True)
    t.start()
    
    # Start Snapshot Syncer
    t_sync = threading.Thread(target=sync_snapshot_file, daemon=True)
    t_sync.start()
    
    # Start Maintenance Task (Pruning)
    t_prune = threading.Thread(target=run_maintenance_tasks, daemon=True)
    t_prune.start()
    
    run_pipeline()
