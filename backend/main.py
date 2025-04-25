import asyncio
from fastapi import FastAPI, Response
import dotenv
from semantic_kernel import Kernel
import uvicorn

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentGroupChat
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy, KernelFunctionTerminationStrategy
from semantic_kernel.contents import ChatHistoryTruncationReducer
from modules.setup.setup_kernel import KernelSetup
from semantic_kernel.functions import KernelFunctionFromPrompt
from plugins.VatsimKnowledgeBasePlugin.VatsimKnowledgeBasePlugin import VatsimKnowledgeBasePlugin
from plugins.VatsimKnowledgeBasePlugin.VatsimKnowledgeBasePluginPhr import VatsimKnowledgeBasePluginPhr
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

    kb_kernel_setup = KernelSetup()
    kb_kernel, kb_kernel_arguments = await kb_kernel_setup.setup_kernel(chat_history, question)

    ph_kernel_setup = KernelSetup()
    ph_kernel, ph_kernel_arguments = await ph_kernel_setup.setup_kernel(chat_history, question)

    ch_kernel_setup = KernelSetup()
    ch_kernel, ch_kernel_arguments = await ph_kernel_setup.setup_kernel(chat_history, question)
    
    kb_kernel.add_plugin(
        VatsimKnowledgeBasePlugin(),
        plugin_name="VatsimKnowledgeBasePlugin",
    )

    ph_kernel.add_plugin(
        VatsimKnowledgeBasePluginPhr(),
        plugin_name="VatsimKnowledgeBasePluginPhraseology",
    )

    kb_agent_name = "knowledgebase_agent"
    ph_agent_name = "phraseology_agent"

    # Agent setup
    knowledgebase_agent = ChatCompletionAgent(
        kernel=kb_kernel, 
        name=kb_agent_name, 
        instructions="""
        Your sole responsibility is to perform a similarity search in the knowledge base and return the results.
        You would respond to any questions about aviation, vatsim like:
        - What is IFR flying
        - What is VFR
        - Provide a map of the VRPs of the Hannover Airport
        
        As a knowledge base agent, you are not allowed to answer any questions about phraseology or aviation phraseology.

        """,
        arguments=kb_kernel_arguments
        )
    
    phraseology_agent = ChatCompletionAgent(
        kernel=ph_kernel, 
        name=ph_agent_name, 
        instructions="""
        Your sole responsibility is to get any IFR or VFR aviation phraseology for the given question.
        Your input would be needed to provide more detailed information about pilots or controllers phraseology elements.
        You would respond to:
        - Which phraseology to use when crossing a CTR?
        - What is the phraseology for flying traffic patterns?
        - How to do the initial call to the controller?
        """,
        arguments=ph_kernel_arguments
        )

    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
        There are two agents: {kb_agent_name} and {ph_agent_name}.
        
        If the user asks for a specific phraseology, the {ph_agent_name} agent should be selected.
        if the user asks about an general aviation question, the {kb_agent_name} agent should be selected.

        If the answer of the {kb_agent_name} agent does include any phraseology, than the {ph_agent_name} agent should refine the phraseology part.

        You must answer with either {kb_agent_name} or {ph_agent_name}.
        History:
        {{{{$history}}}}
        """
    )
    termination_keyword = "yes"
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
        Make sure the users question got answered by the knowedlge base agent.
        If the users question aksed for a specific phraseology, the phraseology agent should have answered it.
        If the question does not ask for a phraseology and the knowledge base agent has answered it, respond with a single word: {termination_keyword}.
        If the question asked for a phraseology and the phraseology agent has answered it, respond with a single word: {termination_keyword}.
        Determine if the phraseology matches the desired request. If so, respond with a single word: {termination_keyword}.

        History:
        {{{{$history}}}}
        """
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=1)

    def result_parser(result):
        if result.value is not None:
            return str(result.value[0]).strip()
        return None

    def termination_parser(result):
        return termination_keyword in str(result.value[0]).lower()


    chat = AgentGroupChat(
        agents=[knowledgebase_agent, phraseology_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=ch_kernel,
            result_parser=result_parser,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            function=termination_function,
            kernel=ch_kernel,
            result_parser=termination_parser,
            agent_variable_name="agents",
            history_variable_name="history",
            maximum_iterations=10
        )
    )
    
    # thread = ChatHistoryAgentThread()

    # history1 = await chat.get_chat_messages(agent=knowledgebase_agent)
    # history2 = await chat.get_chat_messages(agent=phraseology_agent)

    await chat.add_chat_message(message=question)
    try:
        async for response in chat.invoke():
            if response is None or not response.name:
                continue
            print()
            print(f"# {response.name.upper()}:\n{response.content}")
    except Exception as e:
        print(f"Error during chat invocation: {e}")

    # Reset the chat's complete flag for the new conversation round.
    chat.is_complete = False
    return Response(response.content,200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)