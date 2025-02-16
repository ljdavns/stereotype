# Model names
# Model	Anthropic API	AWS Bedrock	GCP Vertex AI
# Claude 3.5 Sonnet	claude-3-5-sonnet-20241022 (claude-3-5-sonnet-latest)	anthropic.claude-3-5-sonnet-20241022-v2:0	claude-3-5-sonnet-v2@20241022
# Claude 3.5 Haiku	claude-3-5-haiku-20241022 (claude-3-5-haiku-latest)	anthropic.claude-3-5-haiku-20241022-v1:0	claude-3-5-haiku@20241022
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
import llm_server_config
from llm.base_llm import BaseLLM, MessageType
import os
import anthropic

total_llm_call_count = 0
total_token_count = 0

MessageTypeDict = {
    MessageType.SYSTEM: 'system',
    MessageType.USER: 'user',
    MessageType.AI: 'assistant'
}

client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class ClaudeLLM(BaseLLM):
    def __init__(self, model_name, meta=None):
        super().__init__(model_name)
        self.meta = meta

    def add_system_instruction(self, instruction):
        self.system_instruction = instruction

    def add_chat_messages(self, messages: list):
        self.chat_history += messages
    
    async def chat(self, new_message, message_type: MessageType, tools=None, chat_history: list = None, **kwargs):
        if chat_history is not None and len(chat_history) > 0:
            self.chat_history = chat_history
        self.chat_history.append({'role': MessageTypeDict[message_type], 'content': new_message})
        
        response = await client.messages.create(
            model=self.model_name,
            messages=self.chat_history,
            system=self.system_instruction,
            max_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0.8)
        )
        
        self.chat_history.append({
            'role': 'assistant',
            'content': response.content[0].text
        })
        
        global total_llm_call_count, total_token_count
        total_llm_call_count += 1
        total_token_count += response.usage.input_tokens + response.usage.output_tokens
        print(f"\n\nTotal LLM call count({self.model_name}): {total_llm_call_count}\nTotal token count({self.model_name}): {total_token_count}\n\n\n")
        
        return True, self.chat_history[-1], self.chat_history, None

async def run_demo():
    llm = ClaudeLLM("claude-3-5-sonnet-latest")
    # llm = ClaudeLLM("claude-3-5-haiku-latest")
    llm.add_system_instruction('You are named Vince in this session')
    succ, ai_message, chat_history, tool_calls = await llm.chat("how are you?", MessageType.USER)
    print("AI Response:", ai_message['content'])
    succ, ai_message, chat_history, tool_calls = await llm.chat("what did we talk about just now?", MessageType.USER)
    print("AI Response:", ai_message['content'])
    succ, ai_message, chat_history, tool_calls = await llm.chat("what is your name in this session?", MessageType.USER)
    print("AI Response:", ai_message['content'])

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_demo())
