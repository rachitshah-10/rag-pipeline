"""LLM interface and Azure AI Foundry OpenAI Responses API client."""

import os
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv


@runtime_checkable
class LLMClient(Protocol):
    """Interface for text generation providers."""

    def generate(self, prompt: str) -> str:
        """Generate a completion string for the given prompt."""
        ...


class AzureFoundryLLMClient:
    """OpenAI SDK client configured for an Azure AI Foundry endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        deployment_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Load Azure Foundry connection settings from arguments or environment variables."""
        load_dotenv()

        self.endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT", "") if endpoint is None else endpoint
        ).strip()
        self.deployment_name = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
            if deployment_name is None
            else deployment_name
        ).strip()
        self.api_key = (
            os.getenv("AZURE_OPENAI_API_KEY", "") if api_key is None else api_key
        ).strip()

        if not self.endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT configuration.")
        if not self.deployment_name:
            raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT configuration.")
        if not self.api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY configuration.")

        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client targeting the Azure Foundry base_url."""
        if self._client is None:
            # Delay OpenAI client construction until live generation so offline tests stay fast.
            from openai import OpenAI

            self._client = OpenAI(base_url=self.endpoint, api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        """Send a prompt via the Responses API and return the generated text."""
        client = self._get_client()
        response = client.responses.create(
            model=self.deployment_name,
            input=prompt,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()
        # Fallback if a response variant exposes structured output items without output_text.
        return str(response.output[0]).strip()
