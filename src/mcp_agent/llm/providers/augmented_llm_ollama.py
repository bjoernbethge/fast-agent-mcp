import os
from typing import List, Optional

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import get_logger

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api"
DEFAULT_OLLAMA_MODEL = "llama3.2"


class OllamaAugmentedLLM(OpenAIAugmentedLLM):
    """
    Augmented LLM implementation for Ollama, a local LLM server.
    
    This implementation leverages Ollama's OpenAI-compatible API to provide
    a consistent interface for using local models with the fast-agent framework.
    
    Ollama-specific features:
        - No API key required for local deployments
        - Custom model loading from Modelfiles
        - Support for various local models (Llama, Mistral, etc.)
    """

    def __init__(self, *args, **kwargs) -> None:
        # Set the provider name before calling parent constructor
        kwargs["provider_name"] = "Ollama"
        self.logger = get_logger(__name__)
        
        # Initialize with parent (OpenAI) implementation
        super().__init__(*args, **kwargs)
        
        # Ollama-specific attributes
        self._model_options = kwargs.get("model_options", {})
        
        model_name = self.default_request_params.model if self.default_request_params else None
        self.logger.info(f"Initializing Ollama LLM with model: {model_name}")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Ollama-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_OLLAMA_MODEL)
        
        # Remove any 'ollama:' prefix if present
        if chosen_model and chosen_model.startswith("ollama:"):
            chosen_model = chosen_model[len("ollama:"):]
        
        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,  # Ollama supports parallel tool calls in compatible models
            max_iterations=10,         # Default for interaction with tools
            use_history=True,          # Maintain conversation history
            temperature=kwargs.get("temperature", 0.7),
            topP=kwargs.get("top_p", 0.9),
            maxTokens=kwargs.get("max_tokens", 2048),
        )

    def _api_key(self) -> Optional[str]:
        """
        Get API key for Ollama.
        
        Ollama typically doesn't require an API key for local deployment,
        but this method allows for authentication in custom setups.
        """
        config = self.context.config
        api_key = None

        # Check config for API key (for remote Ollama instances that might require auth)
        if config and hasattr(config, "ollama") and config.ollama:
            api_key = getattr(config.ollama, "api_key", None)
        
        # Check environment variable
        if api_key is None:
            api_key = os.getenv("OLLAMA_API_KEY")
            
        # Return None is fine for local Ollama setups
        return api_key

    def _base_url(self) -> str:
        """Get the base URL for the Ollama API"""
        base_url = os.getenv("OLLAMA_API_URL", DEFAULT_OLLAMA_BASE_URL)
        
        # Check config file for base URL
        if self.context.config and hasattr(self.context.config, "ollama"):
            config_base_url = getattr(self.context.config.ollama, "base_url", None)
            if config_base_url:
                base_url = config_base_url
                
        # Ensure URL ends with /v1 for OpenAI compatibility
        if not base_url.endswith("/v1"):
            if base_url.endswith("/"):
                base_url = base_url + "v1"
            else:
                base_url = base_url + "/v1"
                
        return base_url
        
    def _get_model_options(self) -> dict:
        """
        Get additional model options for Ollama.
        
        Returns:
            Dictionary of additional options to pass to Ollama
        """
        options = {}
        
        # Add model-specific options from config
        if self.context.config and hasattr(self.context.config, "ollama"):
            ollama_config = self.context.config.ollama
            if hasattr(ollama_config, "model_options"):
                options.update(ollama_config.model_options)
                
        # Add options specified in constructor
        if self._model_options:
            options.update(self._model_options)
            
        return options
        
    async def pre_tool_call(self, tool_call_id: str | None, request):
        """
        Pre-process tool calls for Ollama.
        
        This can be used to adapt tool calls to Ollama's specific requirements.
        For now, we just pass through to the parent implementation.
        """
        self.logger.debug(f"Ollama pre-tool call: {tool_call_id} - {request.params.name}")
        return await super().pre_tool_call(tool_call_id, request)
        
    async def post_tool_call(self, tool_call_id: str | None, request, result):
        """
        Post-process tool call results for Ollama.
        
        This can be used to adapt tool results to Ollama's specific requirements.
        For now, we just pass through to the parent implementation.
        """
        return await super().post_tool_call(tool_call_id, request, result)
