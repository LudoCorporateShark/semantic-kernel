import sys

# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

import asyncio

from samples.getting_started.services import Service
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import (
    ChatHistory,
    # AuthorRole,
    # ImageContent,
    # FinishReason
)

settle
from semantic_kernel.functions import (
    KernelArguments,
    # FunctionResult,
    # KernelFunctionFromMethod,
)


from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

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


# Ensure that this top-level function is marked with 'async'
async def main():
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
        # allow_dangerously_set_content = True # unencoded input, safe content
        execution_settings=execution_settings,
    )

    chat_bot_function = kernel.add_function(  # loading semantic plugin
        function_name="dataAnalysisChatFunc",
        plugin_name="dataAnalysisChatPlugin",
        prompt_template_config=prompt_template_config,
        return_plugin=False,  # switch between plugin (T) and function (F)
        description="Example of chatbot function",
        # fully_qualified_name="dataAnalysisChatPlugin-dataAnalysisChatFunc",  # plugin_name-function_name
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

    # # DEBUG print to see the list of functions and pluging available to the Kernel
    # for plugin_name, plugin in kernel.plugins.items():
    #     for function_name, function in plugin.functions.items():
    #         print(f"Plugin: {plugin_name}, Function: {function_name}")

    # Now keep the chat going:
    async def chat():
        while True:
            input_text = input("You: ")
            if input_text.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
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

    # Start the interactive chat loop
    await chat()


# Call the main function to start the event loop
asyncio.run(main())

# example of inputs:

# input1 = "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"
# input2 = "that sounds interesting, what is it about?"
# input3 = "if I read that book, what exactly will I learn about Greek history?"
# input4 = "could you list some more books I could read about this topic?"
