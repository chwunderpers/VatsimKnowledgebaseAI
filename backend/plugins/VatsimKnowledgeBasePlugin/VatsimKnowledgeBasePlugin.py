from typing import Annotated

from semantic_kernel.functions import kernel_function
from modules.azureopenai.azureopenai import azureopenai
from modules.aisearch.aisearch import aisearch

class VatsimKnowledgeBasePlugin:
    def __init__(self):
        pass
    
    @kernel_function(
            name="SearchKnowledgeBase",
            description="If the user asks a question, the plugin will search the knowledge base and return the results.",
    )
    async def get_knowledgebase_results(self,
        query: Annotated[str,"This is the users raw input question as string"]
        ) -> Annotated[list[dict], "This is the list of results from the knowledge base."]:
        
        embedding =  await azureopenai.create_embedding(query)

        documents = await aisearch.search(embedding, search_text=query)
        return documents