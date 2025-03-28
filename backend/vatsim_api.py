from fastapi import FastAPI, Response
import uvicorn
import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import dotenv
import base64
from pathlib import Path
import json
dotenv.load_dotenv(override=True)

app = FastAPI()

@app.get("/vatsim")
def answer(question: str, history: str = []):
    
    aiseach_client = SearchClient(
    endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_AISEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_AISEARCH_KEY"))
    )

    aoai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    input_embedding = aoai_client.embeddings.create(input=question,model="text-embedding-3-large")
    
    vector_query = VectorizedQuery(
    vector=input_embedding.data[0].embedding,
    k_nearest_neighbors=20,  # Retrieve top 5 similar documents
    fields="contentVector"  # Ensure this matches your index field name
    )
    results = aiseach_client.search(
        search_text="",  # Leave empty for pure vector search
        vector_queries=[vector_query],
        select=["filename", "pageNumber", "content", "type", "path"]  # Adjust based on your index fields
        )
    
    result_final = [res for res in results if res.get("content") is not None]


    messages=[{"role": "system","content": \
               "You are an assistant to answer questions about aviation. \
               Cite where possible with superscript and provide a list of citations including the pageNumber and filename at the end. \
               If there are images (identified by the type = image) inside the context you get, place them to the appropriate location in the text and create a markdown link in the followin format: \
               ![filename](http://localhost:8080/path)\
               Keep caution that there is not a blank space between http: and //.  \
               Format the link like this: ![ebg-3-ged-gin-hef-sig-tau](http://localhost:8080/output_raw_documents/centersektoren/edgg_langen_radar/ebg-3-ged-gin-hef-sig-tau/11_1.png) \
               The answer should always be a proper formatted markdown format."}]
    
    if history:
        messages.extend([msg for msg in json.loads(history)])
    messages.append({"role": "user","content": "user question:" + question + "here is the context you got:" + str(result_final)})

    response = aoai_client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.0
    )
    aoai_client.close()
    aiseach_client.close()

    return Response(response.choices[0].message.content, 200)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)