import asyncio
from fastapi import FastAPI, Response
import dotenv
from semantic_kernel import Kernel
import uvicorn

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent
from modules.setup.setup_kernel import KernelSetup
from plugins.VatsimKnowledgeBasePlugin.VatsimKnowledgeBasePlugin import VatsimKnowledgeBasePlugin
import json

dotenv.load_dotenv(override=True)

app = FastAPI()

@app.get("/vatsim")
async def answer(question: str, history: str = None):
    history = json.loads(history) if history else None
    chat_history = ChatHistory()
    if history:
        for message in history:
            chat_history.add_message(message)

    chat_history.add_system_message("You are an assistant to answer questions about aviation. \
               Cite where possible with superscript and provide a list of citations including the pageNumber and filename at the end. \
               If there are images (identified by the type = image) inside the context you get, place them to the appropriate location in the text and create a markdown link in the followin format: \
               ![filename](http://localhost:8080/path)\
               Keep caution that there is not a blank space between http: and //.  \
               Format the link like this: ![ebg-3-ged-gin-hef-sig-tau](http://localhost:8080/output_raw_documents/centersektoren/edgg_langen_radar/ebg-3-ged-gin-hef-sig-tau/11_1.png) \
               The answer should always be a proper formatted markdown format."
    )

    kernel_setup = KernelSetup()
    kernel, kernel_arguments = await kernel_setup.setup_kernel(chat_history, question)
    
    kernel.add_plugin(
        VatsimKnowledgeBasePlugin(),
        plugin_name="VatsimKnowledgeBasePlugin",
    )

    result = await kernel.invoke(kernel.get_function_from_fully_qualified_function_name("ChatCompletionPlugin-ChatCompletion"), arguments=kernel_arguments)

    return Response(result.value[0].content,200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)