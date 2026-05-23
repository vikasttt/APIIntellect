
"""OpenAI chat completion service (LangChain adapter)."""
from __future__ import annotations
from typing import Any
from app.config.logging import get_logger
logger = get_logger(__name__)
class OpenAIChatService:
    """LangChain-backed async chat with OpenAI."""
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Send messages to OpenAI and return the assistant's reply."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        llm = ChatOpenAI(
            api_key=self._api_key,  # type: ignore[arg-type]
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        lc_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        response = await llm.ainvoke(lc_messages)
        return str(response.content)