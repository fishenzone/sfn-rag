import logging
import httpx
import json
from ..config import settings
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.ollama_base_url = f"http://ollama:{settings.OLLAMA_PORT}"
        self.api_chat_url = f"{self.ollama_base_url}/api/chat"
        logger.info(f"Initializing Ollama client with URL: {self.api_chat_url}")
        self.model_name = settings.OLLAMA_MODEL
        self.timeout = 60

    async def get_completion(self, prompt, max_retries=3):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 16384,
                "num_predict": 1024,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_retries + 1):
                try:
                    logger.info(
                        f"Sending prompt to Ollama (attempt {attempt + 1}/{max_retries + 1})"
                    )

                    response = await client.post(self.api_chat_url, json=payload)
                    response.raise_for_status()

                    try:
                        result = response.json()
                        logger.debug(f"Raw Ollama JSON response: {result}")

                        if "message" in result and "content" in result["message"]:
                            raw_content = result["message"]["content"]
                            logger.debug(f"Raw content from Ollama: {raw_content}")

                            if "</think>" in raw_content:
                                final_content = raw_content.split("</think>")[-1].strip()
                            else:
                                final_content = raw_content.strip()

                            logger.info(f"Parsed content: {final_content}")
                            return final_content
                        else:
                            logger.warning(
                                "Ollama response missing 'message' or 'content' field."
                            )
                            logger.warning(f"Full response: {result}")

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {e}")
                        logger.error(f"Response text that failed to parse: {response.text[:500]}")

                except httpx.RequestError as e:
                    logger.error(f"Ollama request failed: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during Ollama request: {e}")


                if attempt < max_retries:
                    sleep_time = 2**attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        logger.error("All Ollama attempts failed after multiple retries.")
        return "I apologize, but I'm currently unable to generate a response. Please try again later."

