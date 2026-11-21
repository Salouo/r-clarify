from collections.abc import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def convert_chat_messages(messages: Iterable[BaseMessage]) -> list[dict[str, str]]:
    """Convert message roles and text blocks to the format used by both backends."""
    payload: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        payload.append({"role": role, "content": _content_to_text(message.content)})
    return payload
