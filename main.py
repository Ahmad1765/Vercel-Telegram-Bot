import re
from fuzzywuzzy import fuzz
from pony.orm import db_session, select
import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger
from pony.orm import Database, Required, db_session, select
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import telebot
from concurrent.futures import ThreadPoolExecutor


BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DOCX_PATH = BASE_DIR / "2.docx"
COUNTRIES = ["eu", "uk", "usa", "canada"]

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai.api_key = OPENAI_API_KEY
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")
EXECUTOR = ThreadPoolExecutor(max_workers=6)
logger.add(BASE_DIR / "bot.log", rotation="10 MB")

db = Database()
db.bind(provider="sqlite", filename=str(BASE_DIR / "stores.db"), create_db=True)

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
        "JEWELERY"  # Note: typo in your data
    ],
    "home": [
        "HOME / FURNITURE",
        "CLOTHING / HOME",
        "HOME GOODS",
        "High-end MATTRESSES"
    ],
    "footwear": [
        "CLOTHING",  # Footwear might be under clothing
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


class Store(db.Entity):
    name = Required(str)
    category = Required(str)
    country = Required(str)
    price_limit = Required(str)
    item_limit = Required(str)
    notes = Required(str)

db.generate_mapping(create_tables=True)

country_data = {}
for c in COUNTRIES:
    p = BASE_DIR / f"{c}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            country_data[c] = json.load(f)
    else:
        logger.warning("Missing JSON for country: %s", c)

@db_session
def populate_db_once():
    if Store.select().count() > 0:
        return
    logger.info("Populating database...")
    for country_key, data in country_data.items():
        ccode = country_key.upper()
        for category in data.get("categories", []):
            cname = category.get("category_name", "Unknown")
            for s in category.get("stores", []):
                Store(
                    name=s.get("store", "Unnamed"),
                    category=cname,
                    country=ccode,
                    price_limit=s.get("price_limit", "N/A"),
                    item_limit=s.get("item_limit", "N/A"),
                    notes=s.get("notes", "N/A"),
                )
        for s in data.get("global_stores_and_services", []):
            Store(
                name=s.get("store", "Unnamed"),
                category="Global Stores and Services",
                country=ccode,
                price_limit=s.get("price_limit", "N/A"),
                item_limit=s.get("item_limit", "N/A"),
                notes=s.get("notes", "N/A"),
            )
    logger.info("DB population complete.")

populate_db_once()

# -------------------------
# IMPROVED PDF CHUNKING
# -------------------------
CHUNKS_PKL = CACHE_DIR / "docx_chunks_v3.pkl"
EMB_NPY = CACHE_DIR / "docx_emb_v3.npy"

def improved_chunk_docx(path: Path, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    Better chunking strategy for DOCX:
    - Uses python-docx to extract text
    - Splits by paragraphs
    - Uses sliding window with overlap
    - Maintains context
    """
    if CHUNKS_PKL.exists():
        logger.info("Loading cached DOCX chunks")
        return pickle.load(open(CHUNKS_PKL, "rb"))
    
    logger.info("Chunking DOCX with improved strategy...")
    from docx import Document
    
    doc = Document(str(path))
    full_text = ""
    
    # Extract all text from paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            full_text += para.text.strip() + "\n"
    
    # Split into meaningful paragraphs (not just single lines)
    paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 20]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last part for overlap
            words = current_chunk.split()
            overlap_text = " ".join(words[-20:])  # Simplification for overlap
            current_chunk = overlap_text + " " + para
        else:
            current_chunk += " " + para
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [c for c in chunks if len(c) > 50]
    
    pickle.dump(chunks, open(CHUNKS_PKL, "wb"))
    logger.info(f"Created {len(chunks)} DOCX chunks with overlap")
    return chunks

# Load DOCX chunks
if DOCX_PATH.exists():
    pdf_chunks = improved_chunk_docx(DOCX_PATH)
    logger.info(f"Loaded {len(pdf_chunks)} DOCX chunks")
else:
    pdf_chunks = []
    logger.warning(f"DOCX not found at {DOCX_PATH}")

# -------------------------
# IMPROVED EMBEDDINGS
# -------------------------
if pdf_chunks:
    if EMB_NPY.exists():
        pdf_embeddings = np.load(str(EMB_NPY))
        logger.info(f"Loaded PDF embeddings from cache: {pdf_embeddings.shape}")
    else:
        logger.info("Generating embeddings for PDF chunks...")
        pdf_embeddings = []
        batch_size = 50
        
        for i in range(0, len(pdf_chunks), batch_size):
            batch = pdf_chunks[i:i+batch_size]
            try:
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                for item in response.data:
                    emb = np.array(item.embedding, dtype=np.float32)
                    emb /= np.linalg.norm(emb)
                    pdf_embeddings.append(emb)
                
                logger.info(f"Processed {len(pdf_embeddings)}/{len(pdf_chunks)} chunks")
            except Exception as e:
                logger.exception(f"Failed embedding batch {i}: {e}")
        
        pdf_embeddings = np.stack(pdf_embeddings)
        np.save(str(EMB_NPY), pdf_embeddings)
        logger.info(f"Generated and cached PDF embeddings: {pdf_embeddings.shape}")
else:
    pdf_embeddings = None

CATEGORIES = ["ELECTRONICS", "CLOTHING", "JEWELRY"]

def classify_category(query: str) -> str:
    """Classify query into simplified category"""
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

@db_session
def check_store_exists(store_name: str, country: str) -> bool:
    """Check if a specific store exists in the database"""
    normalized_name = store_name.lower().strip()
    stores = select(s for s in Store if s.country == country)[:]
    
    for store in stores:
        if normalized_name in store.name.lower() or store.name.lower() in normalized_name:
            return True
    return False

def parse_json_response(text: str) -> dict:
    """
    Cleans and parses model output that may contain markdown code fences.
    Example input:
      ```json
      { "store_name": "amazon", "category": "electronics" }
      ```
    """
    if not text:
        return {}

    # Remove code fences and leading/trailing whitespace
    cleaned = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover from minor issues like trailing commas
        try:
            cleaned = cleaned.replace("\n", " ").replace(", }", "}")
            return json.loads(cleaned)
        except Exception:
            return {}
        
def extract_store_name(query: str) -> str:
    prompt = f"""
You are an assistant that extracts **store or brand names** from user shopping messages.

Rules:
- Return only the actual store or brand name
- If the message is about a general category (like electronics, clothes, shoes, etc.)
  and does NOT mention a specific store or brand, return null.
- Do not include generic words like "store", "shop", "website", or "online".
- Output JSON exactly like:
  {{"store_name": "Amazon"}}
  or
  {{"store_name": null}}

Message: "{query}"
"""
    response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information when asked."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    response = response.choices[0].message.content.strip()
    data = parse_json_response(response)
    return data.get("store_name") or data.get("category") or ""

@db_session
def fetch_stores_by_mapped_categories(simple_category: str, country: str):
    """Fetch stores using category mapping with case-insensitive matching"""
    if simple_category not in CATEGORY_MAPPING:
        return []
    
    db_categories = CATEGORY_MAPPING[simple_category]
    all_stores = []
    
    # Get all stores for this country
    all_country_stores = select(s for s in Store if s.country == country)[:]
    
    for store in all_country_stores:
        store_category_lower = store.category.lower()
        for db_cat in db_categories:
            # Case-insensitive partial matching
            if db_cat.lower() in store_category_lower or store_category_lower in db_cat.lower():
                store_info = f"{store.name} (Price Limit: {store.price_limit}, Item Limit: {store.item_limit})"
                if store_info not in all_stores:  # Avoid duplicates
                    all_stores.append(store_info)
                break  # Found a match, no need to check other categories
    
    logger.info(f"Found {len(all_stores)} stores for category '{simple_category}' in {country}")
    return all_stores

@db_session
def fetch_stores_for(category: str, country: str):
    return [
        f"{s.name} (Price Limit: {s.price_limit}, Item Limit: {s.item_limit})"
        for s in select(s for s in Store if s.category == category and s.country == country)
    ]

# -------------------------
# IMPROVED SEMANTIC SEARCH
# -------------------------
def semantic_search(query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[float, str]]:
    """
    Improved semantic search with:
    - Better similarity threshold
    - More results for context
    - Score filtering
    """
    if not pdf_chunks or pdf_embeddings is None:
        return []

    try:
        # Get query embedding
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        q_emb = np.array(response.data[0].embedding, dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb)

        # Cosine similarity
        sims = np.dot(pdf_embeddings, q_emb)
        
        # Filter by threshold and get top_k
        valid_indices = np.where(sims > similarity_threshold)[0]
        if len(valid_indices) == 0:
            logger.warning(f"No chunks above similarity threshold {similarity_threshold}")
            # Get top results anyway
            top_indices = sims.argsort()[-top_k:][::-1]
        else:
            # Sort valid indices by score
            sorted_valid = valid_indices[sims[valid_indices].argsort()[::-1]]
            top_indices = sorted_valid[:top_k]

        results = [(float(sims[i]), pdf_chunks[i]) for i in top_indices]
        logger.info(f"Semantic search for '{query}' found {len(results)} results (scores: {[f'{s:.3f}' for s, _ in results]})")
        return results

    except Exception as e:
        logger.exception(f"Semantic search failed: {e}")
        return []

# -------------------------
# IMPROVED RAG ANSWER GENERATION
# -------------------------
def ask_openai_with_context(query: str, contexts: List[str]) -> str:
    """
    Improved RAG with better prompt engineering:
    - Clear instructions
    - Better context formatting
    - Source attribution
    """
    try:
        # Format contexts with numbering
        formatted_context = ""
        for i, ctx in enumerate(contexts, 1):
            formatted_context += f"[Context {i}]:\n{ctx}\n\n"
        
        system_msg = """You are a highly strictly controlled assistant. You MUST follow these rules:
1. ONLY use information provided in the CONTEXTS sections.
2. DO NOT use your own internal knowledge, external websites, or reference any other place.
3. If the answer is not contained within the provided CONTEXTS, explicitly state: "I couldn't find information about that in the available documents."
4. DO NOT make up any information.
5. Be concise and direct.
6. Mention specifically that the information is from the provided document.
"""
        
        user_msg = f"""I will provide you with specific contexts from a document. You MUST answer the question using ONLY these contexts.

CONTEXTS:
{formatted_context}

QUESTION: {query}
"""

        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,  # Lower for more consistent answers
            max_tokens=500,
        )
        
        answer = resp.choices[0].message.content.strip()
        logger.info(f"Generated answer of length {len(answer)}")
        return answer

    except Exception as e:
        logger.exception(f"OpenAI failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# -------------------------
# QUERY CLASSIFICATION
# -------------------------
def classify_query_ai(text: str) -> str:
    """Classify query as 'store' or 'info'"""
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
        # Fallback to keyword matching
        store_keywords = ["store", "buy", "shop", "website", "link", "where can i get"]
        if any(k in text.lower() for k in store_keywords):
            return "store"
        return "info"

def rank_and_summarize_stores(user_query: str, category: str, country: str, stores: list) -> str:
    """Uses OpenAI to rank and summarize relevant stores"""
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

@db_session
def get_all_db_categories(country: str) -> List[str]:
    """Get unique categories for a country"""
    categories = select(s.category for s in Store if s.country == country)[:]
    return list(set(categories))

@db_session
def match_stores_fuzzy(simple_category: str, country: str) -> List[str]:
    """Match stores with category mapping and improved fuzzy fallback"""
    stores = fetch_stores_by_mapped_categories(simple_category, country)
    if stores:
        return stores

    # Fuzzy fallback on actual database categories
    all_db_categories = get_all_db_categories(country)
    if not all_db_categories:
        logger.warning(f"No categories found for country {country}")
        return []
    
    # Try fuzzy matching on database categories
    matched_stores = []
    for db_cat in all_db_categories:
        score = fuzz.partial_ratio(simple_category.lower(), db_cat.lower())
        if score > 50:  # Lowered threshold for better matching
            cat_stores = select(s for s in Store if s.category == db_cat and s.country == country)[:]
            for s in cat_stores:
                store_info = f"{s.name} (Price Limit: {s.price_limit}, Item Limit: {s.item_limit})"
                if store_info not in matched_stores:
                    matched_stores.append(store_info)
    
    if matched_stores:
        logger.info(f"Fuzzy matched {len(matched_stores)} stores for '{simple_category}' in {country}")
        return matched_stores
    
    # Last resort: return all stores if nothing matches
    logger.warning(f"No fuzzy matches for '{simple_category}', returning top stores from all categories")
    all_stores = select(s for s in Store if s.country == country).limit(20)[:]
    return [f"{s.name} (Price Limit: {s.price_limit}, Item Limit: {s.item_limit})" for s in all_stores]

# -------------------------
# BOT HANDLERS
# -------------------------
user_country = {}

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "Welcome! Please set your country: EU, UK, USA, or CANADA")

@bot.message_handler(func=lambda m: True)
def all_messages(message):
    EXECUTOR.submit(handle_user_message, message)

def handle_user_message(message):
    try:
        chat_id = message.chat.id
        text = (message.text or "").strip()
        logger.info(f"Received from {chat_id}: '{text}'")

        # Set country
        if text.upper() in ["EU", "UK", "USA", "CANADA"]:
            user_country[chat_id] = text.upper()
            logger.info(f"Set country for {chat_id} to {text.upper()}")
            bot.reply_to(message, f"✅ Country set to {text.upper()}. How can I help?")
            return

        if chat_id not in user_country:
            logger.warning(f"Country not set for user {chat_id}")
            bot.reply_to(message, "Please set your country first: EU, UK, USA, or CANADA")
            return

        country = user_country[chat_id]
        
        # Classify query type
        query_type = classify_query_ai(text)
        logger.info(f"Query type: {query_type}")

        if query_type.lower() == "store":
            # Check if user is asking about a specific store
            potential_store_name = extract_store_name(text)
            logger.info(f"Potential store name: '{potential_store_name}'")
            
            if potential_store_name:
                store_exists = check_store_exists(potential_store_name, country)
                logger.info(f"Store '{potential_store_name}' exists: {store_exists}")
                
                if not store_exists:
                    # Store doesn't exist - send custom message
                    reply = (
                        f"❌ **{potential_store_name.title()}** is not currently available in our store list.\n\n"
                        "However, we accept **custom store requests**! While we can't guarantee it will work, "
                        "we'll do our best to help you complete your order successfully.\n\n"
                        "🔄 **Alternative Options:**\n"
                        "You can also explore similar stores that we do offer.\n\n"
                        "📝 **Next Steps:**\n"
                        "Please provide a description of the items you're looking to shop for, "
                        "and I'll recommend alternative stores that match your needs."
                    )
                    bot.reply_to(message, reply)
                    return
            
            # Proceed with normal store search
            simple_category = classify_category(text)
            logger.info(f"Classified as simple category: {simple_category}")
            
            try:
                stores = match_stores_fuzzy(simple_category, country)
                logger.info(f"Found {len(stores)} stores for category '{simple_category}' in {country}")
                bot.send_chat_action(chat_id, "typing")
                
                if stores:    
                    logger.info("Ranking and summarizing stores")
                    ranked = rank_and_summarize_stores(text, simple_category, country, stores)
                    reply = f"🏬 *{simple_category.title()}* stores in *{country}*:\n\n{ranked}"
                    bot.reply_to(message, reply)
                else:
                    logger.warning(f"No stores found for category '{simple_category}' in {country}")
                    reply = (
                        f"❌ No stores found in the **{simple_category.title()}** category for **{country}**.\n\n"
                        "We accept **custom store requests**! While we can't guarantee success, "
                        "we'll try our best to help.\n\n"
                        "📝 Please describe the items you want to shop for, "
                        "and I'll recommend alternative stores."
                    )
                    bot.reply_to(message, reply)
            except Exception as e:
                logger.exception(f"Error while processing store query for '{simple_category}' in {country}: {e}")
                bot.reply_to(message, "⚠️ Something went wrong while fetching stores. Please try again later.")

        
        else:
            # Handle INFO queries with improved RAG
            logger.info("Handling INFO query")
            bot.send_chat_action(chat_id, "typing")
            
            results = semantic_search(text, top_k=5, similarity_threshold=0.3)
            logger.info(f"Found {len(results)} relevant chunks")
            
            if results:
                contexts = [chunk for score, chunk in results]
                scores_str = ", ".join([f"{score:.3f}" for score, _ in results])
                logger.info(f"Similarity scores: {scores_str}")
                
                answer = ask_openai_with_context(text, contexts)
                logger.info(f"Generated answer: {answer[:100]}...")
                
                bot.reply_to(message, f"📘 *Answer:*\n\n{answer}")
            else:
                logger.warning("No relevant information found")
                bot.reply_to(message, "I couldn't find relevant information in the documents. Please try rephrasing your question.")
    except Exception as e:
        logger.exception(f"Error handling message from {message.chat.id}: {e}")
        bot.reply_to(message, "⚠️ An error occurred while processing your request. Please try again later.")

# -------------------------
# RUN BOT
# -------------------------
def start_bot():
    logger.info("🤖 Bot starting...")
    try:
        print("Bot info:", bot.get_me())
        bot.infinity_polling()
    except KeyboardInterrupt:
        EXECUTOR.shutdown(wait=False)
        logger.info("Bot stopped.")

if __name__ == "__main__":
    start_bot()