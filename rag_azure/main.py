import os
import logging
import asyncio
from typing import List, Tuple, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# ======================
# CONFIG
# ======================
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)

# Azure Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_AI_SEARCH_INDEX")

# ======================
# CLIENT
# ======================
client_openAI = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# ======================
# FastAPI setup
# ======================
app = FastAPI(title="RAG Chat Azure")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# SESSION & HISTORY
# ======================
sessions: Dict[str, List[dict]] = {}
MAX_HISTORY = 10  # max messaggi (user + assistant)


def get_session(session_id: str = None):
    if session_id is None or session_id not in sessions:
        session_id = str(uuid4())
        sessions[session_id] = [
            {"role": "system",
                "content": "You are a helpful assistant using Azure Search context."}
        ]
    return session_id, sessions[session_id]

# ======================
# UTIL
# ======================


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(v - vmin)/(vmax-vmin) for v in values]


def retrieve_context(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    logging.info("Generazione embedding della query...")
    embedding_response = client_openAI.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    query_vector = embedding_response.data[0].embedding

    # ricerca testuale
    text_top = max(top_k*3, top_k)
    textual_results = search_client.search(
        search_text=query,
        top=text_top,
        include_total_count=False
    )
    textual_map = {}
    for result in textual_results:
        try:
            doc_id = result.get("id") or result.get("Id") or result.get(
                "document_name") or result.get("title")
        except:
            doc_id = None
        if not doc_id:
            continue
        textual_map[doc_id] = {
            "id": doc_id,
            "title": result.get("title", ""),
            "extracted_text": result.get("extracted_text", ""),
            "text_score": float(result.get("@search.score", 0.0))
        }

    # ricerca vettoriale
    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=top_k, fields="embedding")
    vector_results = search_client.search(
        search_text=None, vector_queries=[vector_query], top=top_k)
    vector_map = {}
    for result in vector_results:
        try:
            doc_id = result.get("id") or result.get("Id") or result.get(
                "document_name") or result.get("title")
        except:
            doc_id = None
        if not doc_id:
            continue
        vector_map[doc_id] = {
            "id": doc_id,
            "title": result.get("title", ""),
            "extracted_text": result.get("extracted_text", ""),
            "vector_score": float(result.get("@search.score", 0.0))
        }

    # combinazione
    all_ids = set(list(textual_map.keys()) + list(vector_map.keys()))
    combined_list, text_scores, vector_scores = [], [], []
    for doc_id in all_ids:
        text, vector = textual_map.get(doc_id, {}), vector_map.get(doc_id, {})
        text_score = float(text.get("text_score", 0.0))
        vector_score = float(vector.get("vector_score", 0.0))
        title = text.get("title") or vector.get("title") or ""
        snippet = text.get("extracted_text") or vector.get(
            "extracted_text") or ""
        combined_list.append({
            "id": doc_id,
            "title": title,
            "extracted_text": snippet,
            "text_score": text_score,
            "vector_score": vector_score
        })
        text_scores.append(text_score)
        vector_scores.append(vector_score)

    if not combined_list:
        return "No relevant information found in Azure Search.", []

    text_norm = min_max_normalize(text_scores)
    vec_norm = min_max_normalize(vector_scores)
    for idx, item in enumerate(combined_list):
        item["text_norm"] = text_norm[idx] if idx < len(text_norm) else 0.0
        item["vector_norm"] = vec_norm[idx] if idx < len(vec_norm) else 0.0
        item["hybrid_score"] = 0.5*item["text_norm"] + 0.5*item["vector_norm"]

    combined_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_results = combined_list[:top_k]

    contexts = []
    for r in top_results:
        snippet = r["extracted_text"].strip()
        if r["title"]:
            contexts.append(f"{r['title']}\n{snippet}")
        else:
            contexts.append(snippet)

    full_context = "\n\n".join(contexts)
    return full_context, []


def build_prompt(history: List[dict], context: str, user_input: str) -> List[dict]:
    messages = history.copy()
    messages.append({
        "role": "system",
        "content": f"Here is relevant context retrieved from Azure AI Search (use ONLY this context):\n{context}\n\nAnswer the user's question only using this context. Keep it concise (max 3 sentences)."
    })
    messages.append({"role": "user", "content": user_input})
    return messages


def chat_with_model(messages: List[dict]) -> str:
    response = client_openAI.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content

# ======================
# ROUTES
# ======================


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def api_chat(payload: dict):
    try:
        message = payload.get("message")
        top_k = int(payload.get("top_k", 5))
        session_id = payload.get("session_id")

        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        session_id, history = get_session(session_id)
        context, _ = await asyncio.to_thread(retrieve_context, message, top_k)
        messages = build_prompt(history, context, message)
        answer = await asyncio.to_thread(chat_with_model, messages)

        # aggiorna history lato server
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
            sessions[session_id] = history

        return {"answer": answer, "context": context, "session_id": session_id}
    except Exception as e:
        logging.exception("Errore endpoint /api/chat")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}
