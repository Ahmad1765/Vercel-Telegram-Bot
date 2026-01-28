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
        system_msg = """You are a highly strictly controlled assistant. You MUST follow these rules:
1. ONLY use information provided in the CONTEXTS sections.
2. DO NOT use your own internal knowledge, external websites, or reference any other place.
3. If the answer is not contained within the provided CONTEXTS, explicitly state: "I couldn't find information about that in the available documents."
4. DO NOT make up any information.
5. Be concise and direct.
6. Mention specifically that the information is from the provided document."""
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
    prompt = f"Classify this user query into one of two categories: STORE (user wants recommendations/links) or INFO (user wants info/policies).\n\nQuery: \"{text}\"\n\nAnswer with only: STORE or INFO"
    try:
        resp = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a classifier. Answer only with STORE or INFO."}, {"role": "user", "content": prompt}], max_tokens=5, temperature=0)
        ans = resp.choices[0].message.content.strip().upper()
        return "store" if "STORE" in ans else "info"
    except Exception:
        return "store" if any(k in text.lower() for k in ["store", "buy", "shop", "website"]) else "info"

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
    prompt = f"Rank these most relevant stores (max 10) for query '{user_query}' in category '{category}', country '{country}':\n\n{store_text}"
    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You filter and rank stores intelligently."}, {"role": "user", "content": prompt}], max_tokens=500, temperature=0.4)
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
                ranked = rank_and_summarize_stores(text, cat, country, stores)
                bot.reply_to(message, f"🏬 {cat.title()} stores in {country}:\n\n{ranked}")
            else:
                bot.reply_to(message, f"❌ No stores found for {cat} in {country}.")
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

# --- Flask Routes ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

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
                ranked = rank_and_summarize_stores(message, cat, country, stores)
                reply = f"🏬 {cat.title()} stores in {country}:\n\n{ranked}"
            else:
                reply = f"❌ No stores found."
        else:
            results = semantic_search(message)
            reply = ask_openai_with_context(message, [c for _, c in results]) if results else "I couldn't find info."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
