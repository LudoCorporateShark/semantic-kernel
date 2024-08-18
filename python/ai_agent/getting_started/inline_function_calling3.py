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

prompt = """
{{$input}}

Give me the TLDR in 5 words or less.
"""

input_text = """
    1) A robot may not injure a human being or, through inaction,
    allow a human being to come to harm.

    2) A robot must obey orders given it by human beings except where
    such orders would conflict with the First Law.

    3) A robot must protect its own existence as long as such protection
    does not conflict with the First or Second Law.
"""

prompt_template_config = PromptTemplateConfig(  # semantic plugin configuration
    template=prompt,  # skprompt (plugin) template
    name="tldr",  # skprompt (plugin) name
    template_format="semantic-kernel",  # 'semantic-kernel', 'jinja2' or 'handlebars'
    input_variables=[  # prompt (plugin)  variables
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    # allow_dangerously_set_content=True,  # unencoded input, safe content
    execution_settings=req_settings,
)


# # Run your prompt
# # Note: functions are run asynchronously
tldr_function = (
    kernel.add_function(  # Can also add a list of functions with 'add_functions()'
        function_name="tldrFunction",  # the function name to plug-in the prompt
        plugin_name="tldrPlugin",  # the function to plug-in the prompt
        prompt_template_config=prompt_template_config,
        return_plugin=False,  # switch between plugin (T) and function (F)
        description="TLDR too long; didn't read",
        fully_qualified_name="TLDR_Function",
    )
)


async def main():
    sk_tldr_result = await kernel.invoke(
        tldr_function,  # function(s) to execute, added to the kernel; semantic_configuration
        input=input_text,  # prompt: system message + user input
    )

    print(f"tldrOutput :) {sk_tldr_result}")


if __name__ == "__main__":
    asyncio.run(main())
# If running from a jupyter-notebook:
# await main()
