"""
LLM service layer for portfolio question answering.
Owns provider selection and model invocation so chat orchestration stays focused on business flow.
"""
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from app.core.config import settings
from app.core.exceptions import ConfigurationException, LLMException
from app.core.prompts import (
    DM_AI_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_REPLY,
    build_chat_user_prompt,
)


class LLMService:
    """
    Generates final answers from retrieved context using the configured LLM provider.
    """

    def __init__(self, llm_client: Any | None = None):
        """
        Allows the underlying model client to be injected for tests or future providers.
        """
        self._llm_client = llm_client

    def generate_answer(self, question: str, context: str) -> str:
        """
        Produces a portfolio-grounded answer from retrieved context.
        """
        if not context.strip():
            return INSUFFICIENT_CONTEXT_REPLY

        llm_client = self._llm_client or self._create_llm_client()
        prompt = build_chat_user_prompt(question=question, context=context)
        messages = [
            SystemMessage(content=DM_AI_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = llm_client.invoke(messages)
        except Exception as exc:
            logger.exception(
                "LLM invocation failed for provider {} and model {}: {}",
                settings.LLM_PROVIDER,
                settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.GROQ_MODEL,
                exc,
            )
            raise LLMException("Failed to generate a response from the configured LLM provider.") from exc

        answer = self._extract_text(response)
        return answer or INSUFFICIENT_CONTEXT_REPLY

    def _create_llm_client(self) -> Any:
        """
        Creates the configured LLM client.
        """
        if settings.LLM_PROVIDER == "gemini":
            return self._create_gemini_client()
        if settings.LLM_PROVIDER == "groq":
            raise ConfigurationException(
                "The Groq provider has not been implemented yet. Set LLM_PROVIDER=gemini for this phase."
            )
        raise ConfigurationException(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    def _create_gemini_client(self) -> ChatGoogleGenerativeAI:
        """
        Creates the Gemini chat model client with production-oriented defaults.
        """
        if not settings.GEMINI_API_KEY.strip():
            raise ConfigurationException("GEMINI_API_KEY is not configured.")

        return self._build_cached_gemini_client(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    @staticmethod
    @lru_cache(maxsize=4)
    def _build_cached_gemini_client(
        model: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: int,
        max_retries: int,
    ) -> ChatGoogleGenerativeAI:
        """
        Reuses the Gemini client across requests to avoid repeated initialization overhead.
        """
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @staticmethod
    def _extract_text(response: AIMessage | Any) -> str:
        """
        Extracts plain text from LangChain chat responses across string and block formats.
        """
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_blocks: list[str] = []
            for block in content:
                if isinstance(block, str) and block.strip():
                    text_blocks.append(block.strip())
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        text_blocks.append(text.strip())
            return "\n".join(text_blocks).strip()

        return str(content).strip()
