"""LangChain multi-query generation using OpenAI."""
from __future__ import annotations
from app.config.logging import get_logger
logger = get_logger(__name__)
_MULTI_QUERY_PROMPT = """You are an AI assistant helping to improve document retrieval.
Generate {n} different phrasings of the following question to help find relevant documents.
Output only the alternative questions, one per line, without numbering.
Original question: {query}"""
class MultiQueryService:
    """Generate query variants using an LLM for improved recall."""
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model
    async def generate_queries(self, query: str, n: int = 3) -> list[str]:
        """Return n alternative phrasings of the original query."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatOpenAI(
                api_key=self._api_key,  # type: ignore[arg-type]
                model=self._model,
                temperature=0.7,
            )
            prompt = _MULTI_QUERY_PROMPT.format(n=n, query=query)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
            queries = lines[:n]
            logger.debug(
                "Multi-query generated", original=query, alternatives=queries
            )
            return queries
        except Exception as exc:
            logger.warning("Multi-query generation failed, using original", error=str(exc))
            return []