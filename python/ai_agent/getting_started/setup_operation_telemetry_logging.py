# Copyright (c) Microsoft. All rights reserved.
import sys
import asyncio
import logging
# Define the grandparent directory path
grandparent_dir = r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\python"
# Append the grandparent directory to the Python path
sys.path.append(grandparent_dir)

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
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel

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


async def run_plugin(kernel: Kernel, plugin_name: str, service_id: str):
    """Run a plugin with the given service ID."""
    plugin = kernel.get_plugin(plugin_name)

    poem = await kernel.invoke(
        function=plugin["ShortPoem"],
        arguments=KernelArguments(
            input="Write a poem about John Doe.",
            settings={
                service_id: PromptExecutionSettings(service_id=service_id),
            },
        ),
    )
    print(f"Poem:\n{poem}")

    print("\nTranslated poem:")
    async for update in kernel.invoke_stream(
        function=plugin["Translate"],
        arguments=KernelArguments(
            input=poem,
            language="Italian",
            settings={
                service_id: PromptExecutionSettings(service_id=service_id),
            },
        ),
    ):
        print(update[0].content, end="")
    print()


async def run_service(kernel: Kernel, plugin_name: str, service_id: str):
    """Run a service with the given service ID."""
    print(f"================ Running service {service_id} ================")

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(service_id) as current_span:
        try:
            await run_plugin(kernel, plugin_name=plugin_name, service_id=service_id)
        except Exception as e:
            current_span.record_exception(e)
            print(f"Error running service {service_id}: {e}")


async def main():
    # Initialize the kernel
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(service_id="default"))

    # note: using plugins from the samples folder
    SAMPLE_PLUGIN_FOLDER = (
        r"D:\\CodeBase Docs\\New folder\\semantic-kernel\\prompt_template_samples"
    )

    # Add the sample plugin
    plugin = kernel.add_plugin(
        plugin_name="WriterPlugin",
        parent_directory=SAMPLE_PLUGIN_FOLDER,
    )

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("main") as current_span:
        print(f"Trace ID: {current_span.get_span_context().trace_id}")

        # Run the default AI service
        await run_service(kernel, plugin_name=plugin.name, service_id="default")

if __name__ == "__main__":
    asyncio.run(main())
