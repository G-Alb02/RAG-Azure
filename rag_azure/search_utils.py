import json
import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


def upload_to_search_index(documents, search_endpoint, index_name, search_key):
    try:
        search_client = SearchClient(
            endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_key))
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            search_client.upload_documents(batch)
        logging.info(
            f"Caricati {len(documents)} documenti nell'indice '{index_name}'.")
        return True
    except Exception as e:
        logging.error(f"Errore upload documenti: {e}")
        return False


def semantic_search(query, top_k, embedding_client, search_endpoint, index_name, search_key):
    try:
        query_embedding = embedding_client.embeddings.create(
            model="text-embedding-ada-002", input=query).data[0].embedding
        search_client = SearchClient(
            endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_key))
        vector_query = VectorizedQuery(
            vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")

        results = search_client.search(search_text=None, vector_queries=[
                                       vector_query], top=top_k)
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "extracted_text": result["extracted_text"],
                "title": result["title"],
                "score": result.get("@search.score", 0)
            })
        return formatted_results
    except Exception as e:
        logging.error(f"Errore ricerca semantica: {e}")
        return []
