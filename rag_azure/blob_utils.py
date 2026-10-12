import os
import json
import logging
from datetime import datetime, timezone, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions


def upload_file_to_blob(storage_connection_string: str, file_path: str, container_name: str) -> str:
    """
    Carica un file locale (es. PDF) su Azure Blob Storage nel container specificato.
    Restituisce l'URL del blob.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            storage_connection_string)

        # Crea container se non esiste
        try:
            blob_service_client.create_container(container_name)
            logging.info(f"Container '{container_name}' creato.")
        except Exception:
            logging.info(f"Container '{container_name}' già esistente.")

        blob_name = os.path.basename(file_path)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info(
            f"File '{file_path}' caricato su container '{container_name}'.")
        return blob_client.url

    except Exception as e:
        logging.error(f"Errore caricamento file su Blob: {e}")
        raise


def save_to_blob(storage_connection_string: str, pdf_data: dict, pdf_name: str, container_name: str) -> str:
    """
    Salva il contenuto JSON dell'estrazione PDF su Azure Blob Storage.
    Restituisce l'URL del blob.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            storage_connection_string)

        # Crea container se non esiste
        try:
            blob_service_client.create_container(container_name)
            logging.info(f"Container '{container_name}' creato.")
        except Exception:
            logging.info(f"Container '{container_name}' già esistente.")

        data = {
            "pdf_name": pdf_name,
            "extracted_text": pdf_data["extracted_text"],
            "metadata": {
                "page_count": pdf_data["page_count"],
                "extraction_timestamp": pdf_data["extraction_timestamp"],
                "source_url": pdf_data["source_url"],
                "char_count": len(pdf_data["extracted_text"])
            }
        }

        blob_name = f"{pdf_name}_extracted_text.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name)

        blob_client.upload_blob(json.dumps(
            data, ensure_ascii=False, indent=2), overwrite=True)

        logging.info(
            f"File JSON salvato su container '{container_name}' come '{blob_name}'.")
        return blob_client.url

    except Exception as e:
        logging.error(f"Errore salvataggio JSON su Blob Storage: {e}")
        raise


def generate_blob_sas_url(account_name: str, container_name: str, blob_name: str, account_key: str, expiry_hours: int = 1) -> str:
    """
    Genera un SAS token temporaneo per l'accesso sicuro a un blob PDF.
    Restituisce l'URL con SAS valido per expiry_hours ore.
    """
    try:
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
        )
        return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

    except Exception as e:
        logging.error(f"Errore generazione SAS URL: {e}")
        raise
