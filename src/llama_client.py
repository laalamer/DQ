import requests
from config import OLLAMA_URL, LLAMA_MODEL, REQUEST_TIMEOUT


def ask_llama(prompt):
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    return result.get("response", "")
