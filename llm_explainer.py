from groq import Groq
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_CALL_HISTORY = []
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
COMPARISON_LLM_MODELS = [
    DEFAULT_LLM_MODEL,
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]

SYSTEM_PROMPT = """
You are a careful AutoML analyst. Use only the facts in the user's context.
If a value, metric, feature, probability, or prediction row is not provided, say so.
Never invent dataset meaning, model internals, or prediction confidence.
When explaining predictions, distinguish the predicted value from probability/confidence
and mention that regression outputs are numeric estimates, not class labels.
Be concise, specific, and include the exact numbers that were provided.
"""

def explain(prompt, model=DEFAULT_LLM_MODEL, messages=None):
    start_time = time.perf_counter()
    chat_messages = messages or [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        res = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=0,
            max_tokens=900,
        )
        content = res.choices[0].message.content
        LLM_CALL_HISTORY.append({
            "success": True,
            "model": model,
            "response_time": round(time.perf_counter() - start_time, 3),
            "prompt_length": len(prompt),
            "response_length": len(content),
        })
        return content
    except Exception as e:
        LLM_CALL_HISTORY.append({
            "success": False,
            "model": model,
            "response_time": round(time.perf_counter() - start_time, 3),
            "prompt_length": len(prompt),
            "response_length": 0,
            "error": str(e),
        })
        return "LLM explanation unavailable."


def get_llm_call_history():
    return LLM_CALL_HISTORY.copy()
