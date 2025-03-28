import os
import dotenv
import logging

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

class KernelSetup():
    
    def __init__(self):
        dotenv.load_dotenv()

    async def setup_kernel(self, history, userInput):
        logging.basicConfig(
            format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger("kernel")
        logger.setLevel(logging.INFO)

        # Initialize the kernel
        kernel = Kernel()

        # Add Azure OpenAI chat completion
        kernel.add_service(AzureChatCompletion())
        logger.info("Kernel setup complete.")

        
        kernel.add_function(
            plugin_name="ChatCompletionPlugin",
            function_name="ChatCompletion",
            prompt="{{$chat_history}}{{$user_input}}",
            template_format="semantic-kernel"
        )

        # Enable planning
        execution_settings = AzureChatPromptExecutionSettings()
        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        # Get the chat message content from the chat completion service.
        kernel_arguments = KernelArguments(
            settings=execution_settings,
            # Use keyword arguments to pass the chat history and user input to the kernel function.
            chat_history=history,
            user_input=userInput,
        )

        return kernel, kernel_arguments