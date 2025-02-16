import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import llm_server_config  # ensure configuration is setup
from llm.base_llm import BaseLLM, MessageType
import time
import asyncio

from mistralai import Mistral

# Map the MessageType enum to strings expected by the APIs.
MessageTypeDict = {
    MessageType.SYSTEM: 'system',
    MessageType.USER: 'user',
    MessageType.AI: 'assistant'
}

# Global counters for tracking calls and token usage.
total_llm_call_count = 0
total_token_count = 0
rps_limit = 0.3  # Requests per second limit
last_request_time = 0

class MistralLLM(BaseLLM):
    def __init__(self, model_name, meta=None):
        super().__init__(model_name)
        self.meta = meta
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment.")
        # Create the Mistral client instance.
        self.client = Mistral(api_key=self.api_key)
        # Initialize chat history (if BaseLLM does not already do so).
        self.chat_history = []

    def add_system_instruction(self, instruction):
        """
        Adds or replaces a system instruction at the start of the chat history.
        """
        if not self.chat_history:
            self.chat_history.append({
                'role': MessageTypeDict[MessageType.SYSTEM],
                'content': instruction
            })
        else:
            self.chat_history[0] = {
                'role': MessageTypeDict[MessageType.SYSTEM],
                'content': instruction
            }
    
    def add_chat_messages(self, messages: list):
        """
        Append a list of chat messages to the current chat history.
        """
        self.chat_history += messages

    async def chat(
        self,
        new_message,
        message_type: MessageType,
        tools=None,
        chat_history: list = None,
        **kwargs
    ):
        global total_llm_call_count, total_token_count, last_request_time

        # Rate limiting check
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < 1/rps_limit:  # Less than 1 second since last request
            wait_time = min(1/rps_limit - time_since_last_request, 2)
            print(f"RPS limit exceeded. Waiting for {wait_time:.1f} seconds.")
            await asyncio.sleep(wait_time)
        
        last_request_time = time.time()

        # Optionally refresh chat history if a custom history is provided.
        if chat_history is not None and chat_history:
            self.chat_history = chat_history

        # Add the new message to the chat history.
        self.chat_history.append({
            'role': MessageTypeDict[message_type],
            'content': new_message
        })

        # Call the Mistral asynchronous chat endpoint.
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=self.chat_history,
            max_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0.8),
            stream=False
        )

        # Process the assistant's reply.
        assistant_msg_obj = response.choices[0].message
        if hasattr(assistant_msg_obj, "model_dump"):
            # Use model_dump() if available (as in your OpenAI integration).
            assistant_msg = assistant_msg_obj.model_dump()
        else:
            assistant_msg = {
                'role': MessageTypeDict[MessageType.AI],
                'content': assistant_msg_obj.content
            }
        self.chat_history.append(assistant_msg)

        total_llm_call_count += 1
        # Update tokens if the response contains usage info.
        if hasattr(response, "usage") and hasattr(response.usage, "total_tokens"):
            total_token_count += response.usage.total_tokens

        print(
            f"\n\nTotal LLM call count({self.model_name}): {total_llm_call_count}"
            f"\nTotal token count({self.model_name}): {total_token_count}\n\n"
        )
        return True, assistant_msg, self.chat_history, getattr(assistant_msg_obj, "tool_calls", None)

async def run_demo():
    llm = MistralLLM("mistral-large-latest")
    llm.add_system_instruction('You are named Vince in this session')
    succ, ai_message, chat_history, tool_calls = await llm.chat("how are you?", MessageType.USER)
    print("AI Response:", ai_message.content)
    succ, ai_message, chat_history, tool_calls = await llm.chat("what did we talk about just now?", MessageType.USER)
    print("AI Response:", ai_message.content)
    succ, ai_message, chat_history, tool_calls = await llm.chat("what is your name?", MessageType.USER)
    print("AI Response:", ai_message.content)

if __name__ == '__main__':
    asyncio.run(run_demo())
    pass