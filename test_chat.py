
import os
import sys
from unittest.mock import MagicMock

# Mock libraries to avoid installation issues
sys.modules['telebot'] = MagicMock()
sys.modules['flask'] = MagicMock()
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.ext.declarative'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()

# Set env vars for testing if not present
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-test-key"

# Import functions to test
try:
    from web_app import classify_query_ai, handle_general_chat
except ImportError:
    # If imports fail due to other dependencies, we might need to mock more or adjust
    # For now, let's see if we can import just the functions we need by reading the file
    pass

import openai

# Mock OpenAI for testing logic flow (we don't want to make real calls in this auto-test unless capable)
# However, to test effectively, we might want to check the prompts being sent.
# Let's mock openai.chat.completions.create

original_create = openai.chat.completions.create

def mock_create(*args, **kwargs):
    messages = kwargs.get('messages', [])
    user_content = messages[-1]['content']
    
    # Mock classification
    if "Classify this user query" in user_content:
        if "hello" in user_content.lower() or "hi" in user_content.lower():
            return MagicMock(choices=[MagicMock(message=MagicMock(content="CHAT"))])
        if "store" in user_content.lower():
            return MagicMock(choices=[MagicMock(message=MagicMock(content="STORE"))])
        return MagicMock(choices=[MagicMock(message=MagicMock(content="INFO"))])
    
    # Mock chat response
    if "You are a helpful assistant" in messages[0]['content']:
        return MagicMock(choices=[MagicMock(message=MagicMock(content="Hello there! How can I help you today?"))])
        
    return MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked response"))])

openai.chat.completions.create = mock_create

def test_classification():
    print("Testing Classification...")
    
    c1 = classify_query_ai("hello")
    print(f"Query: 'hello' -> Classified: {c1} (Expected: chat)")
    if c1 != "chat": print("FAIL"); return

    c2 = classify_query_ai("nike store in usa")
    print(f"Query: 'nike store in usa' -> Classified: {c2} (Expected: store)")
    if c2 != "store": print("FAIL"); return

    c3 = classify_query_ai("what is the return policy")
    print(f"Query: 'what is the return policy' -> Classified: {c3} (Expected: info)")
    if c3 != "info": print("FAIL"); return
    
    print("Classification Tests PASSED")

def test_chat_handler():
    print("\nTesting Chat Handler...")
    resp = handle_general_chat("hello")
    print(f"Input: 'hello' -> Response: {resp}")
    if not resp: print("FAIL"); return
    print("Chat Handler Test PASSED")

if __name__ == "__main__":
    try:
        test_classification()
        test_chat_handler()
    except Exception as e:
        print(f"Test failed with error: {e}")
