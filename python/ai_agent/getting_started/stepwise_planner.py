import sys

# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

import asyncio
from typing import Annotated
from samples.getting_started.services import Service
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.core_plugins import (
    # ConversationSummaryPlugin,    # Semantic plugin that enables conversations summarization.
    # HttpPlugin,   # A plugin that provides HTTP functionality: GET, POST, DELETE, PUT
    MathPlugin,
    # SessionsPythonTool,
    # TextMemoryPlugin,     # Semantic Text Memory: RECALL, SAVE
    TextPlugin,  # provides a set of functions to manipulate strings
    TimePlugin,
    # WebSearchEnginePlugin,    # provides web search engine functionality
)
from semantic_kernel.functions import (
    KernelFunctionFromPrompt,
    # kernel_function_decorator,
)
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.planners import (
    FunctionCallingStepwisePlanner,  # can execute multiple API calls with the model - suited for complex tasks that require interconnected steps and dynamic decision-making
    FunctionCallingStepwisePlannerOptions,
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


#  Defining a sample EmailPlugin that simulates handling a request to get_email_address() and send_email().
class EmailPlugin:
    """
    Description: EmailPlugin provides a set of functions to send emails.

    Usage:
        kernel.import_plugin_from_object(EmailPlugin(), plugin_name="email")

    Examples:
        {{email.SendEmail}} => Sends an email with the provided subject and body.
    """

    @kernel_function(
        name="SendEmail",
        description="Given an e-mail and message body, send an e-email",
    )
    def send_email(
        self,
        subject: Annotated[str, "the subject of the email"],
        body: Annotated[str, "the body of the email"],
    ) -> Annotated[str, "the output is a string"]:
        """Sends an email with the provided subject and body."""

        # Implement the send email logic inside the function definition.

        return f"Email sent with subject: {subject} and body: {body}"

    @kernel_function(
        name="GetEmailAddress", description="Given a name, find the email address"
    )
    def get_email_address(
        self,
        input: Annotated[str, "the name of the person"],
    ):
        email = ""
        if input == "Jon":
            email = "emailjon@email.com"
        elif input == "admin":
            email = "emailadmin@email.com"
        elif input == "Joanne":
            email = "emailjoanne@email.com"
        else:
            email = "email@email.com"
        return email


# Ensure that this top-level function is marked with 'async'
async def main():
    # Adding plugins to the Kernel
    kernel.add_plugin(  # from the class EmailPlugin
        plugin_name="EmailPlugin", plugin=EmailPlugin()
    )
    kernel.add_plugin(  # from core plugins
        plugin_name="MathPlugin", plugin=MathPlugin()
    )
    kernel.add_plugin(  # from core plugins
        plugin_name="TimePlugin", plugin=TimePlugin()
    )

    # a set of questions to ask the Planner
    questions = [
        "What is the current hour number, plus 5?",
        "What is 387 minus 22? Email the solution to Ludovic and admin.",
        "Write a limerick, translate it to Spanish, and send it to ludovic",
    ]
    # prompt execution settings to include in the Kernel function calls
    options = FunctionCallingStepwisePlannerOptions(
        max_iterations=10,
        max_tokens=2000,
        max_completion_tokens=500,
        # execution_settings=OpenAIChatPromptExecutionSettings
    )
    # # # Initialise a Stepwise Planner instance with the function call settings - a.k.a "thoughts" and "observations" to (actions) execution
    planner = FunctionCallingStepwisePlanner(service_id=service_id, options=options)

    # # # DEBUG print to see the list of functions and pluging available to the Kernel
    for plugin_name, plugin in kernel.plugins.items():
        for function_name, function in plugin.functions.items():
            print(f"Plugin: {plugin_name}, Function: {function_name}")

    # Get the kernel planner to answer all the questions in the list
    async def plan(ask):
        for question in ask:
            result = await planner.invoke(kernel, question)
            print(f"Q: {question}\nA: {result.final_answer}\n")

        # Uncomment the following line to view the planner's process for completing the request
        print(f"Chat history: {result.chat_history}\n")

    await plan(questions)


# Call the main function to start the event loop
asyncio.run(main())
