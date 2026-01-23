"""
Web-based chat interface for testing the bot locally.
Uses SQLAlchemy instead of Pony ORM for Python 3.14 compatibility.
Run with: python web_app.py
Then open http://localhost:5000 in your browser.
"""
import re
from fuzzywuzzy import fuzz
import os
import json
from pathlib import Path
from typing import List
import openai
from dotenv import load_dotenv
from loguru import logger
from flask import Flask, render_template_string, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).parent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai.api_key = OPENAI_API_KEY
logger.add(BASE_DIR / "web_bot.log", rotation="10 MB")

# SQLAlchemy setup - compatible with Python 3.14
engine = create_engine(f'sqlite:///{BASE_DIR / "stores_new.db"}', echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Store(Base):
    __tablename__ = 'stores'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    country = Column(String, nullable=False)
    price_limit = Column(String, nullable=False)
    item_limit = Column(String, nullable=False)
    notes = Column(String, nullable=False)

Base.metadata.create_all(engine)

CATEGORY_MAPPING = {
    "electronics": [
        "ELECTRONICS / HIGHLY RESELLABLE ITEMS",
        "ELECTRONICS & HIGH RESELL STORES",
        "ELECTRONICS / HIGH RESELL ITEMS",
        "🎮 ELECTRONICS & HIGH RESELL STORES 💸"
    ],
    "clothing": [
        "CLOTHING / HOME",
        "CLOTHING",
        "NETHERLANDS CLOTHING",
        "FRANCE CLOTHING"
    ],
    "jewelry": [
        "JEWELRY",
        "JEWELERY"
    ],
    "home": [
        "HOME / FURNITURE",
        "CLOTHING / HOME",
        "HOME GOODS",
        "High-end MATTRESSES"
    ],
    "footwear": [
        "CLOTHING",
        "SPORTS / OUTDOOR"
    ],
    "sports": [
        "SPORTS / OUTDOOR",
        "❄️OUTDOOR / FOOD ❄️"
    ],
    "food": [
        "FOOD",
        "MEAL PLANS",
        "❄️OUTDOOR / FOOD ❄️"
    ]
}

SIMPLE_CATEGORIES = ["electronics", "clothing", "jewelry", "home", "footwear", "sports", "food"]

# Load country data
country_data = {}
COUNTRIES = ["eu", "uk", "usa", "canada"]
for c in COUNTRIES:
    p = BASE_DIR / f"{c}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            country_data[c] = json.load(f)

def populate_db_once():
    session = Session()
    if session.query(Store).count() > 0:
        session.close()
        return
    logger.info("Populating database with SQLAlchemy...")
    for country_key, data in country_data.items():
        ccode = country_key.upper()
        for category in data.get("categories", []):
            cname = category.get("category_name", "Unknown")
            for s in category.get("stores", []):
                store = Store(
                    name=s.get("store", "Unnamed"),
                    category=cname,
                    country=ccode,
                    price_limit=s.get("price_limit", "N/A"),
                    item_limit=s.get("item_limit", "N/A"),
                    notes=s.get("notes", "N/A"),
                )
                session.add(store)
        for s in data.get("global_stores_and_services", []):
            store = Store(
                name=s.get("store", "Unnamed"),
                category="Global Stores and Services",
                country=ccode,
                price_limit=s.get("price_limit", "N/A"),
                item_limit=s.get("item_limit", "N/A"),
                notes=s.get("notes", "N/A"),
            )
            session.add(store)
    session.commit()
    session.close()
    logger.info("DB population complete.")

populate_db_once()

def parse_json_response(text: str) -> dict:
    if not text:
        return {}
    cleaned = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            cleaned = cleaned.replace("\n", " ").replace(", }", "}")
            return json.loads(cleaned)
        except Exception:
            return {}

def classify_category(query: str) -> str:
    prompt = f"""
You are a text classification assistant. Classify the following user query into one of these categories: {', '.join(SIMPLE_CATEGORIES)}.

- electronics: Electronic devices, gadgets, tech items, gaming consoles, high-resell tech
- clothing: Clothes, fashion, apparel, garments
- jewelry: Rings, necklaces, bracelets, watches, accessories
- home: Furniture, home goods, mattresses, home decor
- footwear: Shoes, sneakers, boots, any footwear
- sports: Sports equipment, outdoor gear, athletic items
- food: Food items, groceries, meal plans

Query: "{query}"

Respond only with the category name (lowercase).
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in SIMPLE_CATEGORIES:
            return "other"
        return category
    except Exception as e:
        logger.error(f"Category classification failed: {e}")
        return "other"

def classify_query_ai(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return "info"

    prompt = f"""Classify this user query into one of two categories:

STORE - User wants store/shop recommendations or links to buy products
INFO - User wants information, explanations, or policy details

Examples:
"Find electronics shops" → STORE
"Where to buy shoes?" → STORE
"What is refund god?" → INFO
"Explain VAT policy" → INFO
"How do price limits work?" → INFO

Query: "{text}"

Answer with only: STORE or INFO"""

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classifier. Answer only with STORE or INFO."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0,
        )
        ans = resp.choices[0].message.content.strip().upper()
        return "store" if "STORE" in ans else "info"
    except Exception as e:
        logger.warning(f"AI classify failed: {e}")
        store_keywords = ["store", "buy", "shop", "website", "link", "where can i get"]
        if any(k in text.lower() for k in store_keywords):
            return "store"
        return "info"

def fetch_stores_by_mapped_categories(simple_category: str, country: str) -> List[str]:
    if simple_category not in CATEGORY_MAPPING:
        return []
    
    db_categories = CATEGORY_MAPPING[simple_category]
    all_stores = []
    
    session = Session()
    all_country_stores = session.query(Store).filter(Store.country == country).all()
    session.close()
    
    for store in all_country_stores:
        store_category_lower = store.category.lower()
        for db_cat in db_categories:
            if db_cat.lower() in store_category_lower or store_category_lower in db_cat.lower():
                store_info = f"{store.name} (Price Limit: {store.price_limit}, Item Limit: {store.item_limit})"
                if store_info not in all_stores:
                    all_stores.append(store_info)
                break
    
    logger.info(f"Found {len(all_stores)} stores for category '{simple_category}' in {country}")
    return all_stores

def get_all_db_categories(country: str) -> List[str]:
    session = Session()
    stores = session.query(Store).filter(Store.country == country).all()
    session.close()
    categories = [s.category for s in stores]
    return list(set(categories))

def match_stores_fuzzy(simple_category: str, country: str) -> List[str]:
    stores = fetch_stores_by_mapped_categories(simple_category, country)
    if stores:
        return stores

    all_db_categories = get_all_db_categories(country)
    if not all_db_categories:
        logger.warning(f"No categories found for country {country}")
        return []
    
    matched_stores = []
    session = Session()
    
    for db_cat in all_db_categories:
        score = fuzz.partial_ratio(simple_category.lower(), db_cat.lower())
        if score > 50:
            cat_stores = session.query(Store).filter(
                Store.category == db_cat, 
                Store.country == country
            ).all()
            for s in cat_stores:
                store_info = f"{s.name} (Price Limit: {s.price_limit}, Item Limit: {s.item_limit})"
                if store_info not in matched_stores:
                    matched_stores.append(store_info)
    
    if matched_stores:
        session.close()
        logger.info(f"Fuzzy matched {len(matched_stores)} stores for '{simple_category}' in {country}")
        return matched_stores
    
    logger.warning(f"No fuzzy matches for '{simple_category}', returning top stores from all categories")
    all_stores = session.query(Store).filter(Store.country == country).limit(20).all()
    session.close()
    return [f"{s.name} (Price Limit: {s.price_limit}, Item Limit: {s.item_limit})" for s in all_stores]

def rank_and_summarize_stores(user_query: str, category: str, country: str, stores: list) -> str:
    if not stores:
        return ""

    store_text = "\n".join(stores)
    prompt = f"""User query: {user_query}
Category: {category}
Country: {country}

Available stores:
{store_text}

Select and rank the most relevant stores (max 10) with brief explanations.
Format as markdown list. Only for the given list"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You filter and rank stores intelligently."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()

# Flask App
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Store Bot - Local Test</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        }
        .chat-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
        }
        .chat-header p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.85rem;
            margin-top: 5px;
        }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .message.bot {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
            border-bottom-left-radius: 5px;
        }
        .message.bot pre {
            white-space: pre-wrap;
            font-family: inherit;
        }
        .country-selector {
            display: flex;
            gap: 10px;
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.2);
            justify-content: center;
        }
        .country-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
        }
        .country-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .country-btn.active {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .chat-input-container {
            display: flex;
            padding: 15px 20px;
            gap: 10px;
            background: rgba(0, 0, 0, 0.3);
        }
        .chat-input {
            flex: 1;
            padding: 12px 18px;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
            outline: none;
        }
        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        .send-btn {
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        .send-btn:hover {
            transform: scale(1.05);
        }
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: none;
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
            padding: 12px 20px;
            border-radius: 18px;
            color: #e0e0e0;
        }
        .typing-indicator.show { display: block; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🏬 Store Bot</h1>
            <p>Find stores by category in your region</p>
        </div>
        <div class="country-selector">
            <button class="country-btn" data-country="EU">🇪🇺 EU</button>
            <button class="country-btn" data-country="UK">🇬🇧 UK</button>
            <button class="country-btn" data-country="USA">🇺🇸 USA</button>
            <button class="country-btn" data-country="CANADA">🇨🇦 Canada</button>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message bot">Welcome! Please select your country above, then ask me about stores.</div>
        </div>
        <div class="typing-indicator" id="typing">Bot is typing...</div>
        <div class="chat-input-container">
            <input type="text" class="chat-input" id="userInput" placeholder="Ask about stores..." disabled>
            <button class="send-btn" id="sendBtn" disabled>Send</button>
        </div>
    </div>

    <script>
        let selectedCountry = null;
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const typingIndicator = document.getElementById('typing');
        
        document.querySelectorAll('.country-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.country-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedCountry = btn.dataset.country;
                userInput.disabled = false;
                sendBtn.disabled = false;
                addMessage(`Country set to ${selectedCountry}. How can I help you?`, 'bot');
            });
        });

        function addMessage(text, type) {
            const msg = document.createElement('div');
            msg.className = `message ${type}`;
            msg.innerHTML = `<pre>${text}</pre>`;
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text || !selectedCountry) return;
            
            addMessage(text, 'user');
            userInput.value = '';
            sendBtn.disabled = true;
            typingIndicator.classList.add('show');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, country: selectedCountry })
                });
                const data = await response.json();
                addMessage(data.reply, 'bot');
            } catch (error) {
                addMessage('Error: Could not get response', 'bot');
            }
            
            typingIndicator.classList.remove('show');
            sendBtn.disabled = false;
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    country = data.get('country', 'EU')
    
    logger.info(f"Received: '{message}' for country: {country}")
    
    try:
        query_type = classify_query_ai(message)
        logger.info(f"Query type: {query_type}")
        
        if query_type == "store":
            simple_category = classify_category(message)
            logger.info(f"Category: {simple_category}")
            
            stores = match_stores_fuzzy(simple_category, country)
            
            if stores:
                ranked = rank_and_summarize_stores(message, simple_category, country, stores)
                reply = f"🏬 {simple_category.title()} stores in {country}:\n\n{ranked}"
            else:
                reply = f"❌ No stores found in the {simple_category.title()} category for {country}.\n\nPlease try a different category or describe what you're looking for."
        else:
            reply = "ℹ️ For information queries, please ask about specific stores or categories. I can help you find:\n\n• Electronics stores\n• Clothing shops\n• Jewelry stores\n• Home goods\n• Sports equipment\n• Food items"
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"reply": f"⚠️ An error occurred: {str(e)}"})

if __name__ == "__main__":
    print("🌐 Starting web server at http://localhost:5000")
    print("📝 Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
