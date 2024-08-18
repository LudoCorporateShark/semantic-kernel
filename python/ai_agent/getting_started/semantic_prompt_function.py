import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

# Initialise Kernel
kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
service_id = "chat-gpt"

# Selecting OpenAI Azure:
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,
    )
)

# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0
req_settings.frequency_penalty = 0
req_settings.presence_penalty = 0

# semantic prompt function #
# # summary prompt function

# Create a reusable function summarize function


prompt_template_config = PromptTemplateConfig(  # semantic plugin configuration
    # template=prompt,  # will be passed in the function call as input
    name="summarize",  # plugin name
    template_format="semantic-kernel",  # 'jinja', etc.
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=req_settings,
)
summarize = kernel.add_function(
    function_name="summarizeFunc",
    plugin_name="summarizePlugin",
    prompt="{{$input}}\n\nSummarize the content above..",  # prompt (input_text) as a function parameter
    prompt_template_settings=req_settings,
)


async def main():
    sk_summary = await kernel.invoke(
        # function(s) to execute, added to the kernel; semantic_configuration
        summarize,
        # prompt: system message + user input
        input="""
        1st Law of Thermodynamics - Energy cannot be created or destroyed.
        2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
        3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.""",
    )
    print(f"\n\nSummaryOutput1: \t{sk_summary}")

    sk_summary = await kernel.invoke(
        # function(s) to execute, added to the kernel; semantic_configuration
        summarize,
        # prompt: system message + user input
        input="""
        1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
        2. The acceleration of an object depends on the mass of the object and the amount of force applied.
        3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first.
        """,
    )
    print(f"\n\nSummaryOutput2: \t{sk_summary}")

    sk_summary = await kernel.invoke(
        # function(s) to execute, added to the kernel; semantic_configuration
        summarize,
        # prompt: system message + user input
        input="""
        Every point mass attracts every single other point mass by a force acting along the line intersecting both points.
        The force is proportional to the product of the two masses and inversely proportional to the square of the distance between them.
        """,
    )
    print(f"\n\nSummaryOutput3: \t{sk_summary}")


if __name__ == "__main__":
    asyncio.run(main())
