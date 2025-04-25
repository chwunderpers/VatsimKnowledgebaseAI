import requests
import json
import os
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


class aisearch():
    def __init__(self):
        pass

    async def search(question_embedding: list[float], search_text: str = "", filter: str = None):
        aiseach_client = SearchClient(
            endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_AISEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_AISEARCH_KEY"))
        )

        results = await aiseach_client.search(
            search_text=search_text,
            filter=filter,
            top=10,
            vector_queries=[
                VectorizedQuery(
                    vector=question_embedding,
                    k_nearest_neighbors=5,
                    fields="contentVector"
                )
            ],
            select=["filename", "pageNumber", "content", "type", "path"]
            )
        
        result_final = [res async for res in results if res.get("content") is not None]

        await aiseach_client.close()

        return result_final