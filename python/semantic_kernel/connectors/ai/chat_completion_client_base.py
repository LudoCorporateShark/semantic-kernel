# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator

from semantic_kernel.services.ai_service_client_base import AIServiceClientBase

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent


class ChatCompletionClientBase(AIServiceClientBase, ABC):
    @abstractmethod
    async def get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> list["ChatMessageContent"]:
        """
        This is the method that is called from the kernel to get a response from a chat-optimized LLM.

        Arguments:
            chat_history {ChatHistory} -- A list of chats in a chat_history object, that can be
                rendered into messages from system, user, assistant and tools.
            settings {PromptExecutionSettings} -- Settings for the request.
            kwargs {Dict[str, Any]} -- The optional arguments.

        Returns:
            Union[str, List[str]] -- A string or list of strings representing the response(s) from the LLM.
        """
        pass

    @abstractmethod
    def get_streaming_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncGenerator[list["StreamingChatMessageContent"], Any]:
        """
        This is the method that is called from the kernel to get a stream response from a chat-optimized LLM.

        Arguments:
            chat_history {ChatHistory} -- A list of chat chat_history, that can be rendered into a
                set of chat_history, from system, user, assistant and function.
            settings {PromptExecutionSettings} -- Settings for the request.
            kwargs {Dict[str, Any]} -- The optional arguments.


        Yields:
            A stream representing the response(s) from the LLM.
        """
        ...

    def _prepare_chat_history_for_request(
        self,
        chat_history: "ChatHistory",
        role_key: str = "role",
        content_key: str = "content",
    ) -> list[dict[str, str | None]]:
        """
        Prepare the chat history for a request, allowing customization of the key names for role/author,
        and optionally overriding the role.

        ChatRole.TOOL messages need to be formatted different than system/user/assistant messages:
            They require a "tool_call_id" and (function) "name" key, and the "metadata" key should
            be removed. The "encoding" key should also be removed.

        Arguments:
            chat_history {ChatHistory} -- The chat history to prepare.

        Returns:
            List[Dict[str, Optional[str]]] -- The prepared chat history.
        """
        return [message.to_dict(role_key=role_key, content_key=content_key) for message in chat_history.messages]
