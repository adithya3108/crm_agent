import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
URL = "https://openrouter.ai/api/v1/chat/completions"


def chat(messages, model=None, temperature=0.0, max_tokens=600, retries=7):
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": model or MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err
