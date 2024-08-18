import sys

# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

import asyncio

from samples.getting_started.services import Service
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel

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
# print(f"Using service type: {selectedService}")

# Remove all services so that this cell can be re-run without restarting the kernel
kernel.remove_all_services()

# # # : 02-running-prompts-from-file
service_id = None
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

print(f"Using service type: {selectedService}")

# note: using plugins from the samples folder
plugins_directory = (
    r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\prompt_template_samples"
)

funFunctions = kernel.add_plugin(
    parent_directory=plugins_directory, plugin_name="FunPlugin"
)

jokeFunction = funFunctions["Joke"]
# print("jokefunction: ", jokeFunction)


async def main():
    result = await kernel.invoke(
        jokeFunction, input="travel to dinosaur age", style="silly"
    )
    print(result)


asyncio.run(main())
