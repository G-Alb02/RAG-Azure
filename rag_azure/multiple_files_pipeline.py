import os
import logging
import concurrent.futures
from dotenv import load_dotenv, find_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

from blob_utils import upload_file_to_blob, save_to_blob, generate_blob_sas_url
from extract_data import (
    extract_data_from_pdf_url,
    extract_data_from_image_url,
    extract_data_from_image_url_with_DI,
)
from index_utils import create_semantic_index
from embedding_utils import (
    initialize_embedding_model,
    split_into_chunks_and_generate_embeddings,
)
from search_utils import upload_to_search_index, semantic_search


def process_file(file_path, storage_connection_string, account_name, account_key,
                 raw_container, extracted_container, DI_client, azure_open_ai_client,
                 embedding_client, search_endpoint, search_key, index_name):
    """Process a single file (PDF or image) through the full RAG pipeline."""
    try:
        logging.info(f"\n🚀 Starting pipeline for file: {file_path}")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_ext = os.path.splitext(file_path)[1].lower()

        # 1️⃣ Upload to RAW container
        blob_url = upload_file_to_blob(
            storage_connection_string, file_path, raw_container)

        # 2️⃣ Generate SAS URL
        sas_url = generate_blob_sas_url(
            account_name, raw_container, os.path.basename(file_path), account_key)

        # 3️⃣ Extract data
        if file_ext == ".pdf":
            extracted_data = extract_data_from_pdf_url(DI_client, sas_url)
        elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            try:
                extracted_data = extract_data_from_image_url(
                    azure_open_ai_client, sas_url)
            except Exception as e:
                logging.warning(
                    f"⚠️ GPT-4o failed on image. Falling back to Document Intelligence: {e}")
                extracted_data = extract_data_from_image_url_with_DI(
                    DI_client, sas_url)

        elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
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

        # 4️⃣ Save extracted JSON to EXTRACTED container
        extracted_json_url = save_to_blob(
            storage_connection_string, extracted_data, file_name, extracted_container)

        # 5️⃣ Generate embeddings & chunking
        documents = split_into_chunks_and_generate_embeddings(
            embedding_client, extracted_data["extracted_text"], file_name, sas_url
        )

        # 6️⃣ Upload chunks to Azure Cognitive Search
        upload_to_search_index(
            documents, search_endpoint, index_name, search_key)

        logging.info(f"✅ Completed processing for {file_name}")
        return {"file": file_name, "status": "success", "blob_url": blob_url, "json_url": extracted_json_url}

    except Exception as e:
        logging.error(f"❌ Error processing {file_path}: {e}")
        return {"file": file_path, "status": "failed", "error": str(e)}


def main():
    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ==========================================================
    # CONFIGURATION
    # ==========================================================
    storage_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

    raw_container = os.environ["AZURE_STORAGE_CONTAINER_NAME_RAW"]
    extracted_container = os.environ["AZURE_STORAGE_CONTAINER_NAME_EXTRACTED"]
    search_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_AI_SEARCH_KEY"]
    index_name = "index-test"

    # 🧠 Initialize clients once
    DI_client = DocumentIntelligenceClient(
        endpoint=os.environ["DOC_INTELLIGENCE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["DOC_INTELLIGENCE_KEY"])
    )
    azure_open_ai_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    )
    embedding_client = initialize_embedding_model()

    # ==========================================================
    # CREATE SEMANTIC INDEX (if needed)
    # ==========================================================
    create_semantic_index(search_endpoint, search_key,
                          index_name, embedding_model_dimension=1536)

    # ==========================================================
    # PROCESS MULTIPLE FILES IN PARALLEL
    # ==========================================================
    input_dir = r"C:\Users\Utente\Desktop\2024-State-of-AI-Change-Readiness-eBook"
    files_to_process = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    ]

    logging.info(f"📂 Found {len(files_to_process)} files to process.")

    results = []
    max_workers = min(4, len(files_to_process))  # Adjust for API rate limits

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_file,
                file_path,
                storage_connection_string,
                account_name,
                account_key,
                raw_container,
                extracted_container,
                DI_client,
                azure_open_ai_client,
                embedding_client,
                search_endpoint,
                search_key,
                index_name,
            ): file_path
            for file_path in files_to_process
        }

        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
            logging.info(f"🧾 Result: {result}")

    # ==========================================================
    #  OPTIONAL: Quick semantic search
    # ==========================================================
    test_queries = ["invoice", "cross country results",
                    "participant list", "map"]
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = semantic_search(
            query, 2, embedding_client, search_endpoint, index_name, search_key)
        for r in results:
            print(f"→ {r['title']} (score {r['score']:.4f})")

    logging.info("\n✅ Full parallel pipeline completed successfully.")
    logging.info(f"📊 Summary: {results}")


if __name__ == "__main__":
    main()
