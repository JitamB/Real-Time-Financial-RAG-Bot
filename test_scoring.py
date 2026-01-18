
import re
from datetime import datetime, timedelta

def format_context(news_tuple, query):
    """Refactored logic from app.py for testing."""
    if not news_tuple:
        return ""
    
    stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 
                  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 
                  'what', 'who', 'when', 'where', 'why', 'how', 'this', 'that'}
    
    import string
    query_words = [
        w.lower().strip(string.punctuation) 
        for w in query.split() 
        if len(w) >= 3 and w.lower().strip(string.punctuation) not in stop_words
    ]
    
    articles = []
    print(f"Query: {query}")
    print(f"Keywords: {query_words}")
    
    for item in news_tuple:
        # Parse timestamp
        timestamp_match = re.search(r'\(([^)]+)\)$', item)
        timestamp_str = timestamp_match.group(1) if timestamp_match else None
        
        # Relevance
        item_lower = item.lower()
        match_count = sum(1 for word in query_words if word and word in item_lower)
        
        # Recency
        try:
            if timestamp_str:
                ts = datetime.fromisoformat(timestamp_str)
                # Mock "now" as slightly after the latest article
                now = datetime.now() # In production this is real time
                # For reproducible test, let's assume "now" is recent
                hours_ago = (now - ts).total_seconds() / 3600
                recency_score = max(0, 100 - hours_ago)
            else:
                recency_score = 0
                hours_ago = 999
        except:
            recency_score = 0
            hours_ago = 999
        
        # New Scoring Formula: Relevance Dominates
        # Match count * 50 means even 1 keyword match (50) beats perfect recency (30)
        combined_score = (match_count * 50.0) + (recency_score * 0.3)
        
        print(f"Article: {item[:30]}... | Time: {timestamp_str} | Match: {match_count} | Recency: {recency_score:.1f} | Score: {combined_score:.2f}")
        
        articles.append({
            'content': item,
            'score': combined_score
        })
    
    articles.sort(key=lambda x: x['score'], reverse=True)
    return articles[:5]

# Mock Data
now = datetime.now()
old_jpm = now - timedelta(hours=18)
new_aapl = now - timedelta(minutes=5)

data = [
    f"JPMorgan forms new unit: details... ({old_jpm.isoformat()})",
    f"Apple stock rises: details... ({new_aapl.isoformat()})",
    f"Google launches AI: details... ({new_aapl.isoformat()})",
    f"Microsoft cloud grows: details... ({new_aapl.isoformat()})",
    f"Amazon delivery drone: details... ({new_aapl.isoformat()})",
    f"Tesla new car: details... ({new_aapl.isoformat()})"
]

print("--- Testing 'JPMorgan' Query ---")
results = format_context(data, "About JPMorgan")
print("\nTop Articles:")
for r in results:
    print(f"- {r['content'][:50]}... (Score: {r['score']:.2f})")
