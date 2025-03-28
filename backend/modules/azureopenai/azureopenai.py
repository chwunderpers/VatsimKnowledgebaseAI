import requests
import json
import os
from openai import AsyncAzureOpenAI


class azureopenai():
    def __init__(self):
        pass

    async def create_embedding(input: str):
        aoai_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

        input_embedding = await aoai_client.embeddings.create(input=input,model="text-embedding-3-large")
        return input_embedding.data[0].embedding