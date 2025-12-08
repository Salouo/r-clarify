import time
from typing import Any

from openai import RateLimitError


def invoke_with_retry(
    llm: Any,
    messages: Any,
    *,
    max_attempts: int = 6,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
):
    """
    Call an LLM with basic backoff on rate limit errors so long runs can auto-resume.

    This intentionally keeps the dependency surface small instead of pulling in
    another retry helper. Other exceptions are not swallowed.
    """

    attempt = 0
    while True:
        try:
            return llm.invoke(messages)
        except RateLimitError:
            attempt += 1
            if attempt >= max_attempts:
                raise

            sleep_for = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(
                f"[Retry] Rate limit hit, waiting {sleep_for:.2f}s "
                f"before retrying ({attempt}/{max_attempts})."
            )
            time.sleep(sleep_for)
