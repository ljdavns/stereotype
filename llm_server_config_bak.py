import os

if 'LLM_MODEL_NAME' not in os.environ or not os.environ['LLM_MODEL_NAME']:
    # os.environ['LLM_MODEL_NAME'] = 'gemini-1.5-flash'
    # os.environ['LLM_MODEL_NAME'] = 'gemini-2.0-flash'
    os.environ['LLM_MODEL_NAME'] = 'gemini-2.0-flash-exp'
    # os.environ['LLM_MODEL_NAME'] = 'gemini-2.0-flash-lite-preview-02-05'
    # os.environ['LLM_MODEL_NAME'] = 'deepseek-chat' # deepseek api is unstable recently, may result in frequent request failure
    # os.environ['LLM_MODEL_NAME'] = 'gpt-4o-mini'
    # os.environ['LLM_MODEL_NAME'] = 'gpt-4o'
    # os.environ['LLM_MODEL_NAME'] = 'mistral-large-latest'
    # os.environ['LLM_MODEL_NAME'] = 'mistral-medium-latest'
    # os.environ['LLM_MODEL_NAME'] = 'mistral-small-latest'
    # os.environ['LLM_MODEL_NAME'] = 'claude-3-5-sonnet-latest'
    # os.environ['LLM_MODEL_NAME'] = 'claude-3-5-haiku-latest'

if 'FC_LLM_MODEL_NAME' not in os.environ or not os.environ['FC_LLM_MODEL_NAME']:
    os.environ['FC_LLM_MODEL_NAME'] = 'gemini-2.0-flash'

if 'EVAL_LLM_MODEL_NAME' not in os.environ or not os.environ['EVAL_LLM_MODEL_NAME']:
    # os.environ['EVAL_LLM_MODEL_NAME'] = 'gemini-2.0-flash'
    os.environ['EVAL_LLM_MODEL_NAME'] = 'gemini-2.0-pro-exp-02-05'

os.environ['GEMINI_API_KEY'] = 'your_gemini_api_key'
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
os.environ['MISTRAL_API_KEY'] = 'your_mistral_api_key'
os.environ['ANTHROPIC_API_KEY'] = 'your_anthropic_api_key' # claude

if 'deepseek' in os.environ['LLM_MODEL_NAME']: # deepseek api is unstable recently, may result in frequent request failure
    os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com'
    os.environ['OPENAI_API_KEY'] = 'your_deepseek_api_key'