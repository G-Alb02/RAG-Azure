# extract_data.py
import logging
import base64
from datetime import datetime, timezone
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from openai import AzureOpenAI


def _common_metadata(source_url: str, typ: str):
    return {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "source_url": source_url,
        "type": typ
    }


def extract_data_from_pdf_url(DI_client: DocumentIntelligenceClient, pdf_sas_url: str):
    """
    Estrae testo da un PDF usando Azure Document Intelligence (prebuilt-read).
    Restituisce un dict uniforme.
    """
    try:
        poller = DI_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(url_source=pdf_sas_url)
        )
        result = poller.result()
        dictionary_result = result.as_dict()

        extracted_text = result.content or ""
        page_count = len(dictionary_result.get("pages", []))

        pdf_data = {
            "extracted_text": extracted_text,
            "page_count": page_count,
            **_common_metadata(pdf_sas_url, "pdf")
        }
        logging.info(
            f"PDF estratto con successo da {pdf_sas_url} (pages={page_count})")
        return pdf_data

    except Exception as e:
        logging.error(f"Errore nell'estrazione PDF: {e}")
        raise


def extract_data_from_image_url(azure_open_ai_client: AzureOpenAI, image_path: str, prompt: str = None):
    """
    Extracts textual and semantic information from an image using GPT-4o multimodal.
    """
    try:
        user_prompt = prompt or (
            "Extract all meaningful text from this image (OCR), list visible objects, "
            "describe graphs in great detail following and saving value changes point by point, "
            "describe any tables (with rows/columns summary if present), and provide a short structured JSON-like summary"
            "suitable for semantic indexing. Return only a clean textual result (no apologies)."
        )

        # Build multimodal content: gestisce sia URL che file locale (base64)
        if image_path.lower().startswith("http"):
            image_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        else:
            # Read and base64-encode the image (attenzione alla dimensione)
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"}}
            ]

        # Call GPT-4o (vision-capable) via AzureOpenAI SDK
        response = azure_open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that performs OCR and visual understanding of images."},
                {"role": "user", "content": image_content}
            ],
            temperature=0
        )

        # Debug: log raw response utile durante il debug (solo DEBUG level)
        try:
            logging.debug("GPT raw response: %s", getattr(
                response, "model_dump_json", lambda **kw: str(response))())
        except Exception:
            logging.debug("GPT raw response stringified: %s", str(response))

        # Robust extraction of returned text: prefer .message.content (singolare)
        extracted_text = ""
        try:
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                # SDK Azure: choice.message.content è il campo corretto
                if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                    extracted_text = choice.message.content
                # fallback a .text se presente (alcune versioni o formati)
                elif hasattr(choice, "text"):
                    extracted_text = choice.text
                else:
                    extracted_text = str(choice)
            else:
                extracted_text = str(response)
        except Exception as parse_exc:
            logging.error("Errore parsing risposta GPT: %s", parse_exc)
            extracted_text = str(response)

        image_data = {
            "extracted_text": extracted_text.strip(),
            "page_count": 1,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "source_url": image_path,
            "type": "image"
        }

        logging.info(f"✅ Image successfully analyzed from {image_path}")
        return image_data

    except Exception as e:
        logging.error(f"❌ Error extracting image data via Azure OpenAI: {e}")
        raise


def extract_data_from_image_url_with_DI(DI_client: DocumentIntelligenceClient, image_sas_url: str):
    """
    Alternativa: usa Document Intelligence (prebuilt-read) per immagini (OCR).
    Utile come fallback quando vuoi più rigore OCR strutturato.
    """
    try:
        poller = DI_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(url_source=image_sas_url)
        )
        result = poller.result()
        extracted_text = result.content or ""
        dictionary_result = result.as_dict()
        page_count = len(dictionary_result.get("pages", []))

        image_data = {
            "extracted_text": extracted_text,
            "page_count": page_count or 1,
            **_common_metadata(image_sas_url, "image")
        }
        logging.info(
            f"Image (DI) successfully analyzed from {image_sas_url} (pages={page_count})")
        return image_data

    except Exception as e:
        logging.error(
            f"Error extracting image data via Document Intelligence: {e}")
        raise
