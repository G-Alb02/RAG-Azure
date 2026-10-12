import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind, VectorSearchAlgorithmMetric
)


def create_semantic_index(search_endpoint: str, search_key: str, index_name: str,
                          embedding_model_dimension: int = 1536):
    """
    Crea un indice semantico con campo vettoriale in Azure Cognitive Search.
    """
    try:
        index_client = SearchIndexClient(
            endpoint=search_endpoint, credential=AzureKeyCredential(search_key))
        try:
            index_client.delete_index(index_name)
        except Exception:
            pass

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="extracted_text",
                            type=SearchFieldDataType.String, searchable=True),
            SearchableField(
                name="title", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="document_name",
                        type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index",
                        type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="source_url", type=SearchFieldDataType.String),
            SimpleField(name="extraction_timestamp",
                        type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_model_dimension,
                vector_search_profile_name="HnswProfile"
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="Hnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={"m": 4, "efConstruction": 400, "efSearch": 500,
                                "metric": VectorSearchAlgorithmMetric.COSINE}
                )
            ],
            profiles=[VectorSearchProfile(
                name="HnswProfile", algorithm_configuration_name="Hnsw")]
        )

        index = SearchIndex(name=index_name, fields=fields,
                            vector_search=vector_search)
        index_client.create_index(index)
        logging.info(f"Indice semantico '{index_name}' creato con successo.")
        return True

    except Exception as e:
        logging.error(f"Errore creazione indice: {e}")
        return False
