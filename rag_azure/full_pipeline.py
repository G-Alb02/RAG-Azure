# full_pipeline.py
import os
import logging
from dotenv import load_dotenv, find_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

from blob_utils import upload_file_to_blob, save_to_blob, generate_blob_sas_url
from extract_data import (
    extract_data_from_pdf_url,
    extract_data_from_image_url,
    extract_data_from_image_url_with_DI
)
from index_utils import create_semantic_index
from embedding_utils import initialize_embedding_model, split_into_chunks_and_generate_embeddings
from search_utils import upload_to_search_index, semantic_search


def main():
    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ==========================================================
    # CONFIGURAZIONE BASE (assicurati che tutte le variabili esistano nel .env)
    # ==========================================================
    storage_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

    raw_container = os.environ["AZURE_STORAGE_CONTAINER_NAME_RAW"]
    extracted_container = os.environ["AZURE_STORAGE_CONTAINER_NAME_EXTRACTED"]

    # Usa r-strings per Windows paths oppure percorsi cross-platform
    file_path = r"C:\Users\Utente\Desktop\Cross_Country\Cross_Country_1.png"
    document_name = os.path.splitext(os.path.basename(file_path))[0]
    index_name = "index-test"

    # ==========================================================
    # 1️⃣ UPLOAD FILE NEL CONTAINER RAW
    # ==========================================================
    logging.info("Caricamento del file nel container RAW...")
    raw_blob_url = upload_file_to_blob(
        storage_connection_string, file_path, raw_container)

    # ==========================================================
    # 2️⃣ GENERAZIONE SAS URL PER L’ACCESSO SICURO
    # ==========================================================
    logging.info("Generazione SAS URL per il blob...")
    blob_name = os.path.basename(file_path)
    sas_url = generate_blob_sas_url(
        account_name, raw_container, blob_name, account_key, expiry_hours=2)

    # ==========================================================
    # 3️⃣ ESECUZIONE DOCUMENT INTELLIGENCE / GPT-4o (in base all'estensione)
    # ==========================================================
    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_data = None

    if file_extension == ".pdf":
        logging.info(
            "Starting PDF analysis with Azure Document Intelligence...")
        DI_client = DocumentIntelligenceClient(
            endpoint=os.environ["DOC_INTELLIGENCE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["DOC_INTELLIGENCE_KEY"])
        )
        extracted_data = extract_data_from_pdf_url(DI_client, sas_url)

    elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        logging.info("Starting image analysis with Azure OpenAI GPT-4o...")
        azure_open_ai_client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        )

        # Prova prima GPT-4o
        extracted_data = extract_data_from_image_url(
            azure_open_ai_client,
            sas_url,
            prompt="Extract all meaningful text and describe the key visual elements from this image."
        )

        # OPTIONAL: se vuoi un fallback OCR più strutturato usa Document Intelligence
        # DI_client = DocumentIntelligenceClient(
        #     endpoint=os.environ["DOC_INTELLIGENCE_ENDPOINT"],
        #     credential=AzureKeyCredential(os.environ["DOC_INTELLIGENCE_KEY"])
        # )
        # if not extracted_data["extracted_text"].strip():
        #     logging.info("Fallback to Document Intelligence for OCR...")
        #     extracted_data = extract_data_from_image_url_with_DI(DI_client, sas_url)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # ==========================================================
    # 4️⃣ SALVATAGGIO FILE JSON NEL CONTAINER EXTRACTED
    # ==========================================================
    logging.info("Salvataggio del file JSON nel container EXTRACTED...")
    extracted_json_url = save_to_blob(
        storage_connection_string, extracted_data, document_name, extracted_container)

    # ==========================================================
    # 5️⃣ CREAZIONE INDICE SEMANTICO SU AZURE AI SEARCH
    # ==========================================================
    search_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_AI_SEARCH_KEY"]
    create_semantic_index(search_endpoint, search_key,
                          index_name, embedding_model_dimension=1536)

    # ==========================================================
    # 6️⃣ GENERAZIONE EMBEDDING E CHUNKING
    # ==========================================================
    embedding_client = initialize_embedding_model()
    documents = split_into_chunks_and_generate_embeddings(
        embedding_client, extracted_data["extracted_text"], document_name, sas_url
    )

    # ==========================================================
    # 7️⃣ UPLOAD DEI DOCUMENTI NELL’INDICE
    # ==========================================================
    upload_to_search_index(documents, search_endpoint, index_name, search_key)

    # ==========================================================
    # 8️⃣ TEST RICERCA SEMANTICA
    # ==========================================================
    test_queries = ["Account creation", "Azure pricing", "Data extraction"]
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = semantic_search(
            query, 2, embedding_client, search_endpoint, index_name, search_key)
        for r in results:
            print(f"→ {r['title']} (score {r['score']:.4f})")

    logging.info("\n✅ Pipeline completata con successo.")
    logging.info(f"📄 File salvato in RAW container: {raw_blob_url}")
    logging.info(
        f"🧾 JSON salvato in EXTRACTED container: {extracted_json_url}")


if __name__ == "__main__":
    main()
