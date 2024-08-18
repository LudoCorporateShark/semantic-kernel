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
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig

# Initialise Kernel
kernel = Kernel()

# Create a service instance
service_settings = ServiceSettings.create()

# Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = (
    Service.AzureOpenAI
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
print(f"Using service type: {selectedService}")

# Remove all services so that this cell can be re-run without restarting the kernel
kernel.remove_all_services()
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

# # # # : 03-prompt-function-inline
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
        max_tokens=2000,
        temperature=0.7,
        # top_p=0,
        # frequency_penalty=0,
        # presence_penalty=0,
    )


# semantic inputs
prompt = """{{$input}}
Summarize the content above.
"""
input_text = """
Demo (ancient Greek poet)
From Wikipedia, the free encyclopedia
Demo or Damo (Greek: Δεμώ, Δαμώ; fl. c. AD 200) was a Greek woman of the Roman period, known for a single epigram, engraved upon the Colossus of Memnon, which bears her name. She speaks of herself therein as a lyric poetess dedicated to the Muses, but nothing is known of her life.[1]
Identity
Demo was evidently Greek, as her name, a traditional epithet of Demeter, signifies. The name was relatively common in the Hellenistic world, in Egypt and elsewhere, and she cannot be further identified. The date of her visit to the Colossus of Memnon cannot be established with certainty, but internal evidence on the left leg suggests her poem was inscribed there at some point in or after AD 196.[2]
Epigram
There are a number of graffiti inscriptions on the Colossus of Memnon. Following three epigrams by Julia Balbilla, a fourth epigram, in elegiac couplets, entitled and presumably authored by "Demo" or "Damo" (the Greek inscription is difficult to read), is a dedication to the Muses.[2] The poem is traditionally published with the works of Balbilla, though the internal evidence suggests a different author.[1]
In the poem, Demo explains that Memnon has shown her special respect. In return, Demo offers the gift for poetry, as a gift to the hero. At the end of this epigram, she addresses Memnon, highlighting his divine status by recalling his strength and holiness.[2]
Demo, like Julia Balbilla, writes in the artificial and poetic Aeolic dialect. The language indicates she was knowledgeable in Homeric poetry—'bearing a pleasant gift', for example, alludes to the use of that phrase throughout the Iliad and Odyssey.[a][2]
"""

# execution_settings_dict = {execution_settings.service_id: execution_settings}

prompt_template_config = PromptTemplateConfig(  # semantic plugin configuration
    template=prompt,  # plugin prompt
    name="summarize",  # plugin name
    template_format="semantic-kernel",  # 'jinja', etc.
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
    # execution_settings={service_id: execution_settings},
)
# prompt_template_config.add_execution_settings(execution_settings)


summarize = kernel.add_function(  # loading semantic plugin
    function_name="summarizeFunc",
    plugin_name="summarizePlugin",
    prompt_template_config=prompt_template_config,
)


async def main():
    sk_summary = await kernel.invoke(
        summarize,  # function(s) to execute, added to the kernel; semantic_configuration
        input=input_text,  # prompt: system message + user input
    )

    print(f"SummaryOutput: {sk_summary}")


if __name__ == "__main__":
    asyncio.run(main())
