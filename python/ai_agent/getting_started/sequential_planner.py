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

from semantic_kernel.functions import (
    KernelArguments,
    # FunctionResult,
    # KernelFunctionFromMethod,
    KernelFunctionFromPrompt,
)

from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

from semantic_kernel.core_plugins import (
    # ConversationSummaryPlugin,    # Semantic plugin that enables conversations summarization.
    # HttpPlugin,   # A plugin that provides HTTP functionality: GET, POST, DELETE, PUT
    # MathPlugin,
    # SessionsPythonTool,
    # TextMemoryPlugin,     # Semantic Text Memory: RECALL, SAVE
    TextPlugin,  # provides a set of functions to manipulate strings
    # TimePlugin,
    # WebSearchEnginePlugin,    # provides web search engine functionality
)

from semantic_kernel.planners import (
    SequentialPlanner,  #   execute a single call to generate a plan that is meant to be saved before continuing operations; XML-based step-by-step planner
    # FunctionCallingStepwisePlanner,  # Perform the next step of the plan if there is more work to do. Here the model can execute many function call as part of his reasoning
)


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

    # Defining what the Kernel should do
    ask = """
    Tomorrow is Valentine's day. I need to come up with a few short poems.
    She likes Shakespeare so write using his style. She speaks French so write it in French.
    Convert the text to uppercase."""

    # # define the location of plugins the Kernel can use
    plugins_directory = (
        r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\prompt_template_samples"
    )
    # providing the Summarisation and Writting functionality (plugin) from files
    summarize_plugin = kernel.add_plugin(
        plugin_name="SummarizePlugin", parent_directory=plugins_directory
    )
    writer_plugin = kernel.add_plugin(
        plugin_name="WriterPlugin",
        parent_directory=plugins_directory,
    )

    # # providing the TextPlugin functionality from core (plugin) functions
    text_plugin = kernel.add_plugin(
        plugin=TextPlugin(), plugin_name="TextPlugin"
    )  # TextPlugin can Trim(start, end), Uppercase, Lowercase

    # # creating the function to use in the Kernel with Prompt Settings
    # in this instance 'Shakespeare' function (shakespeare_func) is added to the set of plugins of 'WritterPlugin'
    # Which seems a plugin is a set of function that can be defined with a their prompt setting if the function Planner decide to use it.
    shakespeare_func = KernelFunctionFromPrompt(
        function_name="Shakespeare",
        plugin_name="WriterPlugin",
        prompt="""
    {{$input}}

    Rewrite the above in the style of Shakespeare.
    """,
        prompt_execution_settings=OpenAIChatPromptExecutionSettings(
            service_id=service_id,
            max_tokens=600,
            temperature=0.8,
        ),
        description="Rewrite the input in the style of Shakespeare.",
    )
    kernel.add_function(plugin_name="WriterPlugin", function=shakespeare_func)

    # # DEBUG print to see the list of functions and pluging available to the Kernel
    # for plugin_name, plugin in kernel.plugins.items():
    #     for function_name, function in plugin.functions.items():
    #         print(f"Plugin: {plugin_name}, Function: {function_name}")

    # # Sequential Planner, a.k.a decision making and function flow
    planner = SequentialPlanner(kernel, service_id)
    # based on the ask (prompt) execute a reasoning plan with a single API call to the model
    sequential_plan = await planner.create_plan(goal=ask)
    # # DEBUG print of all the step decided by the planner to respond to the ask (prompt)
    print("The plan's steps are:")
    for step in sequential_plan._steps:
        print(
            f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
        )

    # print the response of the sequential planner to the ask
    async def plan():
        result = await sequential_plan.invoke(kernel)
        print(f"\n\nplanner: {result}")

    await plan()


# Call the main function to start the event loop
asyncio.run(main())
