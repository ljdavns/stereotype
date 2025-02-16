import asyncio
import time
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
import llm_server_config
from llm.base_llm import BaseLLM, MessageType
import os
import google.generativeai as genai

total_llm_call_count = 0
total_token_count = 0
rpm_limit = 12
request_count = 0
window_start_time = 0  

MessageTypeDict = {
    MessageType.SYSTEM: 'user',
    MessageType.USER: 'user',
    MessageType.AI: 'model'
}

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class RPMLimitExceededError(Exception):
    pass

class Gemini(BaseLLM):
    def __init__(self, model_name, meta=None):
        super().__init__(model_name)
        self.meta = meta
        self.model = genai.GenerativeModel(self.model_name)

    def add_chat_messages(self, messages: list):
        self.chat_history += messages

    def add_system_instruction(self, instruction):
        self.model = genai.GenerativeModel(self.model_name, system_instruction=instruction)
    
    async def chat(self, new_message, message_type: MessageType, tools=None, chat_history: list = None, **kwargs):
        global request_count, window_start_time
        current_time = time.time()
        
        # Initialize window start time if this is the first request
        if window_start_time == 0:
            window_start_time = current_time
        
        # Check if 60 seconds have passed since window start
        if current_time - window_start_time >= 60:
            # Reset counter and start new window
            request_count = 0
            window_start_time = current_time
        
        if request_count >= rpm_limit:
            # Calculate wait time until the window ends
            wait_time = 60 - (current_time - window_start_time)
            if wait_time > 0:
                print(f"RPM limit exceeded. Waiting for {wait_time:.1f} seconds.")
                await asyncio.sleep(wait_time)
            request_count = 0
            window_start_time = time.time()
        
        request_count += 1

        if chat_history is not None and len(chat_history) > 0:
            self.chat_history = chat_history
        self.chat_history.append({'role': MessageTypeDict[message_type], 'content': new_message})
        gemini_chat_history = [{'role': msg['role'], 'parts': msg['content']} for msg in self.chat_history]
        response = await self.model.generate_content_async(
            contents=gemini_chat_history,
            tools=tools,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.8),
            )
        )
        
        self.chat_history.append({
            'role': MessageTypeDict[MessageType.AI],
            'content': response.parts[0].text
        })
        global total_llm_call_count, total_token_count
        total_llm_call_count += 1
        total_token_count += response.usage_metadata.total_token_count
        print(f"\n\nTotal LLM call count({self.model_name}): {total_llm_call_count}\nTotal token count({self.model_name}): {total_token_count}\n\n\n")
        return True, self.chat_history[-1], self.chat_history, response.parts[0].function_call

if __name__ == '__main__':
    llm = Gemini("gemini-1.5-flash")
    llm.add_system_instruction('You are named Vince in this session')
    succ, ai_message, chat_history, tool_calls = llm.chat("how are you?", MessageType.USER)
    succ, ai_message, chat_history, tool_calls = llm.chat("what did we talk about just now?", MessageType.USER)
    succ, ai_message, chat_history, tool_calls = llm.chat("what is your name?", MessageType.USER)
    pass