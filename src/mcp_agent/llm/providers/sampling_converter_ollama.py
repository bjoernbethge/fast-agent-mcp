from typing import Any, Dict

from mcp.types import (
    PromptMessage,
)
from openai.types.chat import (
    ChatCompletionMessage,
)

from mcp_agent.llm.sampling_format_converter import (
    ProviderFormatConverter,
)
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class OllamaSamplingConverter(ProviderFormatConverter[Dict[str, Any], ChatCompletionMessage]):
    """
    Sampling converter for Ollama.
    
    Since Ollama uses an OpenAI-compatible API, we can largely
    reuse the OpenAI converter implementation. This class exists
    primarily for future Ollama-specific extensions.
    """
    
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> Dict[str, Any]:
        """Convert an MCP PromptMessage to an Ollama-compatible message dict."""
        from mcp_agent.llm.providers.multipart_converter_openai import (
            OpenAIConverter,
        )

        # Ollama uses the same message format as OpenAI
        return OpenAIConverter.convert_prompt_message_to_openai(message)
