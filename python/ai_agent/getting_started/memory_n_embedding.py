import sys

# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

import asyncio

from samples.getting_started.services import Service
from samples.service_settings import ServiceSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import AzureTextEmbedding
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import OpenAITextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.functions import (
    # KernelArguments,
    KernelFunction,
    # FunctionResult,
    # KernelFunctionFromMethod,
)

from semantic_kernel.prompt_template import PromptTemplateConfig #, InputVariable

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

# Chat as service_id
chat_service_id = "chat"

# Configure AI service execution setting to be used by the kernel
if selectedService == Service.AzureOpenAI:
    azure_chat_service = AzureChatCompletion(   # chat settings
        service_id=chat_service_id,
    )
    embedding_gen = AzureTextEmbedding(   # embedding settings
        service_id="embedding",
    )
    # loading chat and embedding settings
    kernel.add_service(azure_chat_service)
    kernel.add_service(embedding_gen)
elif selectedService == Service.OpenAI:
    oai_chat_service = OpenAIChatCompletion(
        service_id=chat_service_id,
    )
    embedding_gen = OpenAITextEmbedding(
        ai_model_id="embedding",
    )
    kernel.add_service(oai_chat_service)
    kernel.add_service(embedding_gen)


# # logging and telemetry # #

# telemetry imports
import logging
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorLogExporter,
    AzureMonitorMetricExporter,
    AzureMonitorTraceExporter,
)
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import DropAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import set_tracer_provider
from samples.demos.telemetry_with_application_insights.telemetry_sample_settings import TelemetrySampleSettings

# Load settings
settings = TelemetrySampleSettings.create()

# Create a resource to represent the service/sample
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "TelemetryExample"})


def set_up_logging():
    log_exporter = AzureMonitorLogExporter(connection_string=settings.connection_string)

    # Create and set a global logger provider for the application.
    logger_provider = LoggerProvider(resource=resource)
    # Log processors are initialized with an exporter which is responsible
    # for sending the telemetry data to a particular backend.
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    # Sets the global default logger provider
    set_logger_provider(logger_provider)

    # Create a logging handler to write logging records, in OTLP format, to the exporter.
    handler = LoggingHandler()
    # Add a filter to the handler to only process records from semantic_kernel.
    handler.addFilter(logging.Filter("semantic_kernel"))
    # Attach the handler to the root logger. `getLogger()` with no arguments returns the root logger.
    # Events from all child loggers will be processed by this handler.
    logger = logging.getLogger()
    logger.addHandler(handler)
    # Set the logging level to NOTSET to allow all records to be processed by the handler.
    logger.setLevel(logging.NOTSET)


def set_up_tracing():
    trace_exporter = AzureMonitorTraceExporter(connection_string=settings.connection_string)

    # Initialize a trace provider for the application. This is a factory for creating tracers.
    tracer_provider = TracerProvider(resource=resource)
    # Span processors are initialized with an exporter which is responsible
    # for sending the telemetry data to a particular backend.
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    # Sets the global default tracer provider
    set_tracer_provider(tracer_provider)


def set_up_metrics():
    metric_exporter = AzureMonitorMetricExporter(connection_string=settings.connection_string)

    # Initialize a metric provider for the application. This is a factory for creating meters.
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
        resource=resource,
        views=[
            # Dropping all instrument names except for those starting with "semantic_kernel"
            View(instrument_name="*", aggregation=DropAggregation()),
            View(instrument_name="semantic_kernel*"),
        ],
    )
    # Sets the global default meter provider
    set_meter_provider(meter_provider)


set_up_logging()
set_up_tracing()
set_up_metrics()
# #

# # Semantic Memory allows also to index external data sources, without duplicating all the information
memory = SemanticTextMemory(
    storage=VolatileMemoryStore(),      # The MemoryStoreBase to use for storage  - # it exist MemoryQueryResult() which work with memory.get(), MemoryRecord() etc for other possibility
    embeddings_generator=embedding_gen  # The EmbeddingGeneratorBase to use for generating embeddings
)

# # Add the plugin to interact with a Semantic Text Memory
kernel.add_plugin(
    TextMemoryPlugin(memory),   # from the selected memory store base
    "TextMemoryPlugin"          # name of the memory plugin
)

# # Semantic Memory is a set of data structures that allow you to store the meaning of text that come from different data sources.
# # The texts are embedded or compressed into a vector of floats representing mathematically the texts' contents and meaning.

# creating some initial memories for the chat flow in sample generic memory
collection_id = "generic"

# prepping the sample memory
async def populate_memory(memory: SemanticTextMemory) -> None:
    # Add some documents to the semantic memory
    await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
    await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
    await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")
    # other functionalities include:
        #  memory.get_collections, to get a specific collection to search in
        #  memory.save_reference, to save a reference into a specific memory
        #  memory.search, to search for information in the memory
        #  memory.get, to get information from the memory, works with MemoryQueryResult() if information found

# populating the sample memory
# await populate_memory(memory)

async def search_memory_examples(memory: SemanticTextMemory) -> None:
    questions = [
        "What is my budget for 2024?",
        "What are my savings from 2023?",
        "What are my investments?",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(collection_id, question)
        print(f"Answer: {result[0].text}\n")

# Main function to run the async functions
async def main():
    # add a trace for logging details
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("main") as current_span:
        print(f"Trace ID: {current_span.get_span_context().trace_id}")

        # Populate memory
        await populate_memory(memory)

        # Search memory examples
        await search_memory_examples(memory)

# Execute the async main function
if __name__ == "__main__":
    asyncio.run(main())
