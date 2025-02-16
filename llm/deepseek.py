from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
import llm_server_config
from llm.base_llm import BaseLLM, MessageType
import os
from openai import AsyncOpenAI

total_llm_call_count = 0
total_token_count = 0

MessageTypeDict = {
    MessageType.SYSTEM: 'system',
    MessageType.USER: 'user',
    MessageType.AI: 'assistant'
}

client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_API_BASE'] if 'OPENAI_API_BASE' in os.environ else None)
class DeepSeek(BaseLLM):
    def __init__(self, model_name, meta=None):
        super().__init__(model_name)
        self.meta = meta

    def add_system_instruction(self, instruction):
        if len(self.chat_history) == 0:
            self.chat_history = [{'role': MessageTypeDict[MessageType.SYSTEM], 'content': instruction}]
        else:
            self.chat_history[0] = {'role': MessageTypeDict[MessageType.SYSTEM], 'content': instruction}

    def add_chat_messages(self, messages: list):
        self.chat_history += messages
    
    async def chat(self, new_message, message_type: MessageType, tools=None, chat_history: list = None, **kwargs):
        # meta = self.meta
        if tools is not None and len(tools) == 0:
            tools = None
        if chat_history is not None and len(chat_history) > 0:
            self.chat_history = chat_history
        self.chat_history.append({'role': MessageTypeDict[message_type], 'content': new_message})
        response = await client.chat.completions.create(
            model=self.model_name,
            tools=tools,
            messages=self.chat_history,
            max_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0.8),
            stream=False
        )
        # print(response.choices[0].message.content)
        # self.chat_history.append({
        #     'role': response.choices[0].message.role,
        #     'content': (response.choices[0].message.content if response.choices[0].message.content is not None else '') + \
        #         ('\n' + str(response.choices[0].message.tool_calls) if response.choices[0].message.tool_calls is not None else '')
        # })
        self.chat_history.append(response.choices[0].message.model_dump())
        global total_llm_call_count, total_token_count
        total_llm_call_count += 1
        total_token_count += response.usage.total_tokens
        print(f"\n\nTotal LLM call count({self.model_name}): {total_llm_call_count}\nTotal token count({self.model_name}): {total_token_count}\n\n\n")
        return True, self.chat_history[-1], self.chat_history, response.choices[0].message.tool_calls
       

if __name__ == '__main__':
    llm = DeepSeek("deepseek-chat")
    llm.add_system_instruction('You are named Vince in this session')
    succ, ai_message, chat_history, tool_calls = llm.chat("how are you?", MessageType.USER)
    succ, ai_message, chat_history, tool_calls = llm.chat("what did we talk about just now?", MessageType.USER)
    succ, ai_message, chat_history, tool_calls = llm.chat("what is your name?", MessageType.USER)
    pass