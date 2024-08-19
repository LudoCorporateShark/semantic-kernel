import sys

# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

import asyncio

from samples.getting_started.services import Service
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel

# FunctionResult,
# KernelFunctionFromMethod,
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import (
    ChatHistory,
)

# AuthorRole,
# ImageContent,
# FinishReason
from semantic_kernel.functions import (
    KernelArguments,
)
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig


# Ensure that this top-level function is marked with 'async'
async def main():

    # Create a service instance
    service_settings = ServiceSettings.create()

    # Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
    selectedService = (
        Service.AzureOpenAI
        if service_settings.global_llm_service is None
        else Service(service_settings.global_llm_service.lower())
    )
    print(f"Using service type: {selectedService}")

    # Initialise Kernel
    kernel = Kernel()

    service_id = None
    # service_id = "AzureOpenAI"  # "default"
    if selectedService == Service.OpenAI:
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

        service_id = "default"
        kernel.add_service(
            OpenAIChatCompletion(
                service_id=service_id,
            ),
        )
    elif selectedService == Service.AzureOpenAI:
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

        service_id = "default"
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
            ),
        )

    # # Initialising AI service
    if selectedService == Service.OpenAI:
        execution_settings = OpenAIChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id="gpt-3.5-turbo",
            max_tokens=2000,
            temperature=0.7,
        )
    elif selectedService == Service.AzureOpenAI:
        execution_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id="gpt4",
            max_tokens=500,
            temperature=0.7,
            # top_p=0,
            # frequency_penalty=0,
            # presence_penalty=0,
        )

    # Create a chat history volatile instance
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful chatbot who is good about giving book recommendations."
    )

    # # semantic inputs
    # chatbot dummy prompt
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if it does not have an answer.

    {{$history}}
    User: {{$user_input}}
    ChatBot:
    """

    # registering the semantic function
    prompt_template_config = PromptTemplateConfig(  # semantic plugin configuration
        template=prompt,  # chat prompt
        name="data-analysis-chat",  # chat name
        template_format="semantic-kernel",  # 'jinja', 'handlebars'
        input_variables=[
            InputVariable(name="input", description="The user input", is_required=True),
            InputVariable(
                name="history", description="The conversation history", is_required=True
            ),
        ],
        execution_settings=execution_settings,
    )

    chat_bot_function = kernel.add_function(  # loading semantic plugin
        function_name="dataAnalysisChatFunc",
        plugin_name="dataAnalysisChatPlugin",
        prompt_template_config=prompt_template_config,
        return_plugin=False,  # switch between plugin (T) and function (F)
        description="Example of chatbot function",
        fully_qualified_name="Chatbot_Function",
    )

    # Create a chat history volatile instance
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful chatbot who is good about giving book recommendations."
    )

    # Initialising the Kernel argument
    arguments = KernelArguments(
        user_input="Hi, I'm looking for book suggestions", history=chat_history
    )

    # The top-level await is replaced with a function call to 'main()'
    # The 'main' function will contain the await statements
    bot_response = await kernel.invoke(
        chat_bot_function,  # function(s) to execute, added to the kernel; semantic_configuration
        arguments,
        # input=input_text,  # chat (conversation) input
    )

    # update the chat history after the initial intro message
    chat_history.add_assistant_message(str(bot_response))
    # other chat capibilites: # ChatHistory.add_message(msg)
    #                                      .restore_chat_history(chat_history_json)
    #                                      .load_chat_history_from_file(file_path)

    # Now keep the chat going:
    async def chat(input_text: str) -> None:
        # Save new message in the context variables
        print(f"User: {input_text}")

        # Process the user message and get an answer
        answer = await kernel.invoke(
            chat_bot_function,
            KernelArguments(user_input=input_text, history=chat_history),
        )

        # Show the response
        print(f"ChatBot: {answer}")

        # continue to update the chat history with the new prompt input
        chat_history.add_user_message(input_text)
        chat_history.add_assistant_message(str(answer))

    await chat(
        "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"
    )


# Call the main function to start the event loop
asyncio.run(main())
