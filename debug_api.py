import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("ALPHA_VANTAGE_KEY")
print(f"Key present: {bool(key)}")

if key:
    symbol = "AAPL"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={key}"
    print(f"Fetching {url}...")
    try:
        r = requests.get(url)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")
