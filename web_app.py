import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger
from flask import Flask, render_template_string, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import telebot
from concurrent.futures import ThreadPoolExecutor

# --- Initialization ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai.api_key = OPENAI_API_KEY
bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)  # Webhooks usually work better without internal threading
EXECUTOR = ThreadPoolExecutor(max_workers=6)

# --- SQLAlchemy setup ---
engine = create_engine(f'sqlite:///{BASE_DIR / "stores_new.db"}', echo=False, connect_args={"check_same_thread": False})
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

# --- RAG Data and Functions ---
DOCX_PATH = BASE_DIR / "2.docx"
CHUNKS_PKL = CACHE_DIR / "docx_chunks_v3.pkl"
EMB_NPY = CACHE_DIR / "docx_emb_v3.npy"

docx_chunks = []
docx_embeddings = None

def load_rag_data():
    global docx_chunks, docx_embeddings
    if CHUNKS_PKL.exists() and EMB_NPY.exists():
        logger.info("Loading cached RAG data...")
        docx_chunks = pickle.load(open(CHUNKS_PKL, "rb"))
        docx_embeddings = np.load(str(EMB_NPY))
    else:
        logger.warning("RAG cache files missing! Bot will not be able to answer info queries.")

load_rag_data()

CATEGORY_MAPPING = {
    "electronics": ["ELECTRONICS / HIGHLY RESELLABLE ITEMS", "ELECTRONICS & HIGH RESELL STORES", "ELECTRONICS / HIGH RESELL ITEMS", "🎮 ELECTRONICS & HIGH RESELL STORES 💸"],
    "clothing": ["CLOTHING / HOME", "CLOTHING", "NETHERLANDS CLOTHING", "FRANCE CLOTHING"],
    "jewelry": ["JEWELRY", "JEWELERY"],
    "home": ["HOME / FURNITURE", "CLOTHING / HOME", "HOME GOODS", "High-end MATTRESSES"],
    "footwear": ["CLOTHING", "SPORTS / OUTDOOR"],
    "sports": ["SPORTS / OUTDOOR", "❄️OUTDOOR / FOOD ❄️"],
    "food": ["FOOD", "MEAL PLANS", "❄️OUTDOOR / FOOD ❄️"]
}
SIMPLE_CATEGORIES = ["electronics", "clothing", "jewelry", "home", "footwear", "sports", "food"]

# --- Helper Functions ---

def semantic_search(query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[float, str]]:
    if not docx_chunks or docx_embeddings is None:
        return []
    try:
        response = openai.embeddings.create(model="text-embedding-3-small", input=query, timeout=30.0)
        q_emb = np.array(response.data[0].embedding, dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb)
        sims = np.dot(docx_embeddings, q_emb)
        valid_indices = np.where(sims > similarity_threshold)[0]
        if len(valid_indices) == 0:
            top_indices = sims.argsort()[-top_k:][::-1]
        else:
            sorted_valid = valid_indices[sims[valid_indices].argsort()[::-1]]
            top_indices = sorted_valid[:top_k]
        return [(float(sims[i]), docx_chunks[i]) for i in top_indices]
    except Exception as e:
        logger.exception(f"Semantic search failed: {e}")
        return []

def ask_openai_with_context(query: str, contexts: List[str]) -> str:
    try:
        formatted_context = "".join([f"[Context {i+1}]:\n{ctx}\n\n" for i, ctx in enumerate(contexts)])
        system_msg = """You are a helpful assistant. You MUST follow these rules:
1. ONLY use information provided in the CONTEXTS sections.
2. DO NOT use your own internal knowledge or external websites.
3. If the answer is not contained within the provided CONTEXTS, simply state that you couldn't find that information.
4. Answer naturally and concisely.
5. DO NOT constantly phrase sentences like "According to the document" or "The context mentions". Just give the answer directly."""
        user_msg = f"I will provide you with specific contexts from a document. You MUST answer the question using ONLY these contexts.\n\nCONTEXTS:\n{formatted_context}\nQUESTION: {query}"
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"OpenAI failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

def classify_query_ai(text: str) -> str:
    prompt = f"Classify this user query into one of three categories: STORE (user wants recommendations/links), INFO (user wants specific info/policies from documents), or CHAT (general greetings, small talk, identity questions, or 'hello/hi').\n\nQuery: \"{text}\"\n\nAnswer with only: STORE, INFO, or CHAT"
    try:
        resp = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a classifier. Answer only with STORE, INFO, or CHAT."}, {"role": "user", "content": prompt}], max_tokens=5, temperature=0)
        ans = resp.choices[0].message.content.strip().upper()
        if "STORE" in ans: return "store"
        if "CHAT" in ans: return "chat"
        return "info"
    except Exception:
        # Fallback heuristics
        text_lower = text.lower()
        if any(k in text_lower for k in ["store", "buy", "shop", "website"]): return "store"
        if any(k in text_lower for k in ["hello", "hi", "hey", "who are you", "what can you do"]): return "chat"
        return "info"

def handle_general_chat(query: str) -> str:
    try:
        system_msg = "You are a helpful assistant for a Store Bot. You help users find stores and provide information. Be friendly and concise. If the user asks what you can do, explain that you can recommend stores and answer questions about policies."
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"General chat failed: {e}")
        return "Hello! I can help you find stores or answer policy questions."

def classify_category(query: str) -> str:
    prompt = f"Classify query into categories: {', '.join(SIMPLE_CATEGORIES)}.\n\nQuery: \"{query}\"\n\nRespond only with category name (lowercase)."
    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}], max_tokens=10, temperature=0)
        category = response.choices[0].message.content.strip().lower()
        return category if category in SIMPLE_CATEGORIES else "other"
    except Exception:
        return "other"

def match_stores_fuzzy(simple_category: str, country: str) -> List[str]:
    session = Session()
    try:
        db_categories = CATEGORY_MAPPING.get(simple_category, [])
        matched_stores = []
        all_country_stores = session.query(Store).filter(Store.country == country.upper()).all()
        for s in all_country_stores:
            for db_cat in db_categories:
                if db_cat.lower() in s.category.lower() or s.category.lower() in db_cat.lower():
                    matched_stores.append(f"{s.name} (Price: {s.price_limit}, Item: {s.item_limit})")
                    break
        if matched_stores: return list(set(matched_stores))
        # Fallback to general fuzzy (simplified for speed)
        return [f"{s.name} (Price: {s.price_limit}, Item: {s.item_limit})" for s in all_country_stores[:10]]
    finally:
        session.close()

def rank_and_summarize_stores(user_query: str, category: str, country: str, stores: list) -> str:
    if not stores: return ""
    store_text = "\n".join(stores)
    prompt = f"Analyze these stores and return a JSON array of the top 10 most relevant ones for query '{user_query}' (Category: {category}, Country: {country}).\n\nReturn ONLY raw JSON (no markdown formatting, no backticks). Each object must have keys: 'name', 'price_limit', 'item_limit', 'approx_time', 'notes'. If a field is missing, use an empty string.\n\nStores:\n{store_text}"
    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON."}, {"role": "user", "content": prompt}], max_tokens=1000, temperature=0.4)
        return response.choices[0].message.content.strip()
    except Exception:
        return store_text

# --- Telegram Logic ---
user_country = {}

@bot.message_handler(commands=['start'])
def start_cmd(message):
    bot.reply_to(message, "Welcome! Please set your country: EU, UK, USA, or CANADA")

@bot.message_handler(func=lambda m: True)
def handle_msg(message):
    try:
        chat_id = message.chat.id
        text = (message.text or "").strip()
        if text.upper() in ["EU", "UK", "USA", "CANADA"]:
            user_country[chat_id] = text.upper()
            bot.reply_to(message, f"✅ Country set to {text.upper()}.")
            return
        if chat_id not in user_country:
            bot.reply_to(message, "Please set your country first: EU, UK, USA, or CANADA")
            return
        
        country = user_country[chat_id]
        query_type = classify_query_ai(text)
        
        if query_type == "store":
            cat = classify_category(text)
            stores = match_stores_fuzzy(cat, country)
            if stores:
                json_str = rank_and_summarize_stores(text, cat, country, stores)
                try:
                    # Clean up if OpenAI adds backticks
                    if json_str.startswith("```"):
                        json_str = json_str.split("```")[1].strip()
                        if json_str.startswith("json"): json_str = json_str[4:].strip()
                    
                    store_data = json.loads(json_str)
                    formatted_reply = f"🏬 *{cat.title()} stores in {country}:*\n\n"
                    for idx, s in enumerate(store_data, 1):
                        formatted_reply += f"{idx}. *{s.get('name', 'Unknown')}*\n"
                        if s.get('price_limit'): formatted_reply += f"   💰 Limit: {s['price_limit']}\n"
                        if s.get('item_limit'): formatted_reply += f"   📦 Items: {s['item_limit']}\n"
                        if s.get('approx_time'): formatted_reply += f"   ⏱️ Time: {s['approx_time']}\n"
                        if s.get('notes'): formatted_reply += f"   📝 {s['notes']}\n"
                        formatted_reply += "\n"
                    bot.reply_to(message, formatted_reply, parse_mode="Markdown")
                except Exception as e:
                    logger.error(f"JSON Parse Error: {e}")
                    # Fallback if JSON fails
                    bot.reply_to(message, f"🏬 {cat.title()} stores in {country} (Raw):\n\n{json_str}")
            else:
                bot.reply_to(message, f"❌ No stores found for {cat} in {country}.")
        elif query_type == "chat":
            reply = handle_general_chat(text)
            bot.reply_to(message, reply)
        else:
            results = semantic_search(text)
            if results:
                answer = ask_openai_with_context(text, [c for _, c in results])
                bot.reply_to(message, f"📘 *Answer:*\n\n{answer}")
            else:
                bot.reply_to(message, "I couldn't find relevant information in the documents.")
    except Exception as e:
        logger.exception(e)
        bot.reply_to(message, "⚠️ Something went wrong.")

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Store Bot - Web Interface</title>
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

        /* Store Card Styles */
        .store-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            width: 100%;
            margin-top: 10px;
        }
        .store-card {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 16px;
            transition: transform 0.2s, background 0.2s;
            animation: fadeIn 0.4s ease;
        }
        .store-card:hover {
            transform: translateY(-3px);
            background: rgba(255, 255, 255, 0.15);
        }
        .store-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 8px;
            margin-bottom: 10px;
        }
        .store-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
        }
        .store-details {
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 0.9rem;
            color: #e0e0e0;
        }
        .detail-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .detail-icon { opacity: 0.7; }
        .store-notes {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.85rem;
            font-style: italic;
            color: rgba(255, 255, 255, 0.7);
        }
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
            if (type === 'bot') {
                msg.innerHTML = text; // Allow HTML for bot
            } else {
                msg.textContent = text;
            }
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function renderStoreList(stores, category, country) {
            const container = document.createElement('div');
            container.className = 'message bot';
            container.style.background = 'transparent';
            container.style.padding = '0';
            
            const title = document.createElement('div');
            title.innerHTML = `<strong>🏬 ${category.toUpperCase()} stores in ${country}:</strong>`;
            title.style.marginBottom = '10px';
            title.style.color = '#e0e0e0';
            title.style.padding = '0 10px';
            container.appendChild(title);

            const grid = document.createElement('div');
            grid.className = 'store-grid';

            stores.forEach(store => {
                const card = document.createElement('div');
                card.className = 'store-card';
                card.innerHTML = `
                    <div class="store-header">
                        <span class="store-name">${store.name || 'Unknown Store'}</span>
                    </div>
                    <div class="store-details">
                        ${store.price_limit ? `<div class="detail-row"><span class="detail-icon">💰</span> <span>Limit: ${store.price_limit}</span></div>` : ''}
                        ${store.item_limit ? `<div class="detail-row"><span class="detail-icon">📦</span> <span>Items: ${store.item_limit}</span></div>` : ''}
                        ${store.approx_time ? `<div class="detail-row"><span class="detail-icon">⏱️</span> <span>Time: ${store.approx_time}</span></div>` : ''}
                    </div>
                    ${store.notes ? `<div class="store-notes">📝 ${store.notes}</div>` : ''}
                `;
                grid.appendChild(card);
            });

            container.appendChild(grid);
            messagesDiv.appendChild(container);
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
                
                if (data.type === 'store_list') {
                    renderStoreList(data.data, data.category, data.country);
                } else {
                    addMessage(data.reply, 'bot');
                }
            } catch (error) {
                console.error(error);
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

# --- Flask Routes ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    else:
        return 'Invalid content type', 403

# Re-using the local chat interface logic from original web_app.py
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    country = data.get('country', 'EU')
    try:
        query_type = classify_query_ai(message)
        if query_type == "store":
            cat = classify_category(message)
            stores = match_stores_fuzzy(cat, country)
            if stores:
                json_str = rank_and_summarize_stores(message, cat, country, stores)
                try:
                    if json_str.startswith("```"):
                        json_str = json_str.split("```")[1].strip()
                        if json_str.startswith("json"): json_str = json_str[4:].strip()
                    store_data = json.loads(json_str)
                    return jsonify({"type": "store_list", "data": store_data, "category": cat, "country": country})
                except:
                    reply = f"🏬 {cat.title()} stores in {country}:\n\n{json_str}"
            else:
                reply = f"❌ No stores found."
        elif query_type == "chat":
             reply = handle_general_chat(message)
        else:
            results = semantic_search(message)
            reply = ask_openai_with_context(message, [c for _, c in results]) if results else "I couldn't find info."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
