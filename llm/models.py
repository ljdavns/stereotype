from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from llm.gemini import Gemini
from llm.gpt import OpenAILLM
from llm.deepseek import DeepSeek
from llm.mistral import MistralLLM
from llm.claude import ClaudeLLM

MODEL_NAME_TO_LLM = {
    'gemini-1.5-flash': Gemini,
    'gemini-1.5-pro': Gemini,
    'gemini-2.0-flash': Gemini,
    'gemini-2.0-flash-exp': Gemini,
    'gemini-2.0-flash-lite-preview-02-05': Gemini,
    'gemini-2.0-pro-exp-02-05': Gemini,
    'gpt-4o': OpenAILLM,
    'gpt-4o-mini': OpenAILLM,
    'deepseek-chat': DeepSeek,
    'mistral-large-latest': MistralLLM,
    'mistral-medium-latest': MistralLLM,
    'mistral-small-latest': MistralLLM,
    'claude-3-5-sonnet-latest': ClaudeLLM,
    'claude-3-5-haiku-latest': ClaudeLLM,
}
