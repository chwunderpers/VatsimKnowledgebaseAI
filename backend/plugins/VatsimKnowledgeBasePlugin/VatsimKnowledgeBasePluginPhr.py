from typing import Annotated

from semantic_kernel.functions import kernel_function
from modules.azureopenai.azureopenai import azureopenai
from modules.aisearch.aisearch import aisearch

class VatsimKnowledgeBasePluginPhr:
    def __init__(self):
        pass
    
    @kernel_function(
            name="PhrKnowledgeBase",
            description="If the user asks for phraseology, this method searches the knowledge base for phraseology examples.",
    )
    async def get_knowledgebase_results_phraseology(self,
        query: Annotated[str,"This is the users raw input question as string"]
        ) -> Annotated[list[dict], "This is the list of results from the knowledge base."]:
        
        embedding =  await azureopenai.create_embedding(query)

        documents = await aisearch.search(embedding, search_text=query, filter="path eq 'output_raw_documents\\atc_training_german\\vfr.pdf'")
        return documents