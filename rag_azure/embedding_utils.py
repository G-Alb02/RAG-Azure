import os
import json
import logging
from datetime import datetime
from openai import AzureOpenAI


def initialize_embedding_model():
    """
    Inizializza il client Azure OpenAI per la generazione di embedding.
    """
    try:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-05-01-preview"
        )
        logging.info(
            "Client Azure OpenAI Embedding inizializzato correttamente.")
        return client
    except Exception as e:
        logging.error(f"Errore inizializzazione embedding model: {e}")
        raise


def generate_embedding(client, text: str):
    """
    Genera un embedding per un testo tramite Azure OpenAI (text-embedding-ada-002).
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Errore generazione embedding: {e}")
        raise


def split_into_chunks_and_generate_embeddings(embedding_client, extracted_text, document_name, source_url):
    """
    Divide il testo in chunk e genera gli embedding con Azure OpenAI.
    """
    try:
        chunk_size = 200
        overlap = 40
        words = extracted_text.split()
        chunks, documents = [], []

        for i in range(0, len(words), chunk_size - overlap):
            text_in_chunk = " ".join(words[i:i + chunk_size - 1])
            if text_in_chunk.strip():
                chunks.append({
                    "text": text_in_chunk.strip(),
                    "index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + chunk_size, len(words))
                })

        for chunk in chunks:
            embedding = generate_embedding(embedding_client, chunk["text"])
            documents.append({
                "id": f"{document_name}_chunk_{chunk['index']}",
                "extracted_text": chunk["text"],
                "title": f"{document_name} - Chunk {chunk['index'] + 1}",
                "document_name": document_name,
                "chunk_index": chunk["index"],
                "source_url": source_url,
                "extraction_timestamp": datetime.now().isoformat(),
                "embedding": embedding
            })
        return documents

    except Exception as e:
        logging.error(f"Errore chunk/embedding: {e}")
        raise
