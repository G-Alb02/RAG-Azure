import os
import logging
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# ======================
# CARICAMENTO VARIABILI DI AMBIENTE
# ======================

load_dotenv()

# --- OpenAI ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)

# --- Azure Search ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_AI_SEARCH_INDEX")

# ======================
# INIZIALIZZAZIONE CLIENT
# ======================

# OpenAI Client (solo embedding + completions/chat)
client_openAI = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

# Search Client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# ======================
# UTIL: normalizzazione min-max
# ======================
"""
Funzione per comparare i risultati dei search score (testuale 
e vettoriale) prima della combinazione
"""


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        # tutti uguali -> restituisci 1.0 per evitare divisione per zero
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

# ======================
# RETRIEVE: RICERCA IBRIDA (COMBINAZIONE PUNTEGGI)
# ======================


def retrieve_context(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    """
    Esegue retrieval ibrido:
      - genera embedding per la query;
      - esegue una ricerca testuale (search_text) e una vettoriale;
      - normalizza i punteggi e combina (peso uguale).
    Restituisce il contesto concatenato (stringa) e una lista vuota (non mostriamo fonti).
    """
    logging.info("Generazione embedding della query...")
    embedding_response = client_openAI.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    query_vector = embedding_response.data[0].embedding

    # --- 1) ricerca testuale (ottieni più risultati per avere più materiale da unire) ---
    logging.info("Esecuzione ricerca testuale su Azure Search...")
    # prendi più risultati testuali per fusione
    text_top = max(top_k * 3, top_k)
    textual_results = search_client.search(
        search_text=query,
        top=text_top,
        include_total_count=False
    )

    # raccogli punteggi testuali e metadata
    textual_map: Dict[str, Dict[str, Any]] = {}
    for result in textual_results:
        try:
            doc_id = result.get("id") or result.get("Id") or result.get(
                "document_name") or result.get("title")
        except Exception:
            doc_id = None
        if not doc_id:
            continue
        textual_map[doc_id] = {
            "id": doc_id,
            "title": result.get("title", ""),
            "extracted_text": result.get("extracted_text", ""),
            "text_score": float(result.get("@search.score", 0.0))
        }

    # --- 2) ricerca vettoriale ---
    logging.info("Esecuzione ricerca vettoriale su Azure Search...")

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    vector_results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        top=top_k
    )

    vector_map: Dict[str, Dict[str, Any]] = {}
    for result in vector_results:
        try:
            doc_id = result.get("id") or result.get("Id") or result.get(
                "document_name") or result.get("title")
        except Exception:
            doc_id = None
        if not doc_id:
            continue
        vector_map[doc_id] = {
            "id": doc_id,
            "title": result.get("title", ""),
            "extracted_text": result.get("extracted_text", ""),
            # per i risultati vettoriali @search.score rappresenta la similarità
            "vector_score": float(result.get("@search.score", 0.0))
        }

    # --- 3) unisci id (union) e costruisci lista con text_score e vector_score ---
    all_ids = set(list(textual_map.keys()) + list(vector_map.keys()))
    combined_list = []
    text_scores = []
    vector_scores = []

    for doc_id in all_ids:
        text = textual_map.get(doc_id, {})
        vector = vector_map.get(doc_id, {})
        text_score = float(text.get("text_score", 0.0))
        vector_score = float(vector.get("vector_score", 0.0))
        title = text.get("title") or vector.get("title") or ""
        extracted_text = text.get("extracted_text") or vector.get(
            "extracted_text") or ""
        combined_list.append({
            "id": doc_id,
            "title": title,
            "extracted_text": extracted_text,
            "text_score": text_score,
            "vector_score": vector_score
        })
        text_scores.append(text_score)
        vector_scores.append(vector_score)

    if not combined_list:
        logging.info("Nessun risultato rilevante trovato in Azure Search.")
        return "No relevant information found in Azure Search.", []

    # --- 4) normalizza i punteggi (min-max) e combinali (50/50) ---
    text_norm = min_max_normalize(text_scores)
    vec_norm = min_max_normalize(vector_scores)

    # associare le normalizzazioni correttamente
    for idx, item in enumerate(combined_list):
        item["text_norm"] = text_norm[idx] if idx < len(text_norm) else 0.0
        item["vector_norm"] = vec_norm[idx] if idx < len(vec_norm) else 0.0
        # combinazione con peso 50/50 (puoi cambiare pesi qui)
        item["hybrid_score"] = 0.5 * \
            item["text_norm"] + 0.5 * item["vector_norm"]

    # --- 5) ordina per punteggio ibrido e prendi top_k ---
    combined_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_results = combined_list[:top_k]

    # --- 6) costruisci contesto concatenato ---
    contexts = []
    for r in top_results:
        # includiamo titolo + testo recuperato. Non mostriamo le fonti secondo le tue istruzioni.
        snippet = r["extracted_text"].strip()
        if r["title"]:
            contexts.append(f"{r['title']}\n{snippet}")
        else:
            contexts.append(snippet)

    # concatenazione separata da due newline (RAG-friendly)
    full_context = "\n\n".join(contexts)
    return full_context, []  # ritorniamo lista fonti vuota perché non vuoi mostrarle

# ======================
# COSTRUZIONE PROMPT
# ======================


def build_prompt(history: List[dict], context: str, user_input: str) -> List[dict]:
    messages = history.copy()
    # sistema: istruzioni che impongono all'assistente di rispondere solo in base al contesto
    messages.append({
        "role": "system",
        "content": (
            f"Here is relevant context retrieved from Azure AI Search (use ONLY this context):\n{context}\n\n"
            "Answer the user's question **only** using the provided context. "
            "If the answer cannot be found in the context, say so clearly. "
            "Keep responses concise (max 3 sentences)."
        )
    })
    messages.append({
        "role": "user",
        "content": user_input
    })
    return messages

# ======================
# CHAT CON IL MODELLO
# ======================


def chat_with_model(messages: List[dict]) -> str:
    response = client_openAI.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content

# ======================
# INTERACTIVE (stesso comportamento CLI)
# ======================


def interactive_chat():
    print("🤖 RAG ChatBot (Hybrid Search: Azure OpenAI + Azure Search) — modalità RAG\nScrivi 'exit' per terminare.\n")

    history = [{
        "role": "system",
        "content": (
            "You are a helpful assistant that uses Azure AI Search context to answer questions accurately. "
            "Only answer questions based on the retrievable data from Azure AI Search."
            "Never get lazy."
        )
    }]

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        # Recupera contesto (ricerca ibrida)
        context, _ = retrieve_context(user_input, top_k=5)

        # Costruisci prompt
        messages = build_prompt(history, context, user_input)

        # Ottieni risposta
        answer = chat_with_model(messages)

        # Mostra risposta
        print(f"\n🤖 Assistant:\n{answer}\n")

        # Aggiorna cronologia
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    #                    format="%(asctime)s [%(levelname)s] %(message)s")
    interactive_chat()
