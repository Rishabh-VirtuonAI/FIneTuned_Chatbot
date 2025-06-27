from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import mysql.connector
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from utils.db_fetcher import get_raw_text_from_db
from utils.formatter import format_text_for_knowledge_base
from utils.vector_builder import build_faiss_index
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(root_path="/chatbot_llm")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cache loaded domain handlers
loaded_domains = {}

# Define your allowed domains
ALLOWED_DOMAINS = [
    "audio", "backlight", "bluetooth_wifi", "camera", "charging", "dead", "graphics",
    "hardware_others", "network", "reprogramming_changing", "sim_sdcard",
    "software_unlock", "touch_fingerprint", "ufs_reprogramming_changing", "usb_flashing"
]

class ChatRequest(BaseModel):
    user_query: str
    chat_history: List[dict] = []
    username: str

def load_domain_handler(domain: str):
    if domain not in ALLOWED_DOMAINS:
        raise HTTPException(status_code=404, detail="Invalid domain")

    if domain in loaded_domains:
        return loaded_domains[domain]

    try:
        module = importlib.import_module(f"domains.{domain}.handler")
        loaded_domains[domain] = module
        return module
    except ModuleNotFoundError:
        raise HTTPException(status_code=500, detail="Handler not found for domain")

@app.post("/chat/{domain}")
async def chat(domain: str, request: ChatRequest):
    handler = load_domain_handler(domain)
    return await handler.chat_with_user(request.user_query, request.chat_history,request.username)



# knowledge base updation part
@app.post("/update-kb/data_{domain}")
async def update_kb(domain: str):
    if domain not in ALLOWED_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Invalid domain: {domain}")

    logger.info(f"Updating knowledge base for domain: {domain}")

    # Step 1: Fetch raw text from DB
    raw_text = get_raw_text_from_db(domain)

    # Step 2: Format using LLM
    formatted_text = format_text_for_knowledge_base(raw_text)

    # Step 3: Save formatted KB to file
    kb_path = f"domains/{domain}/knowledge_base.txt"
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    with open(kb_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    # Step 4: Build FAISS index
    build_faiss_index(domain, kb_path)

    return {"status": "success", "message": f"Knowledge base updated for domain '{domain}'"}