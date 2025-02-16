import json
import copy
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from llm.base_llm import MessageType, BaseLLM
from agent.base_agent import BaseAgent
import traceback
import time
from json_repair import repair_json
import asyncio

parser_example = """
    User: "**Performance Analysis:**

- **person_0 (me)**:
  - Successful in: Manager (2x), Data Scientist (2x), Janitor, Rehabilitation Counselor, Truck Driver (2x)
  - Failed in: None
  - Best suited role: Manager, Data Scientist, or Truck Driver

- **person_1**:
  - Successful in: Truck Driver (2x), Data Scientist (2x), Manager (2x), Rehabilitation Counselor
  - Failed in: Janitor (2x)
  - Best suited role: Manager, Data Scientist, or Truck Driver

- **person_2**:
  - Successful in: Rehabilitation Counselor (3x), Manager (2x), Janitor, Data Scientist, Truck Driver
  - Failed in: Data Scientist
  - Best suited role: Rehabilitation Counselor or Manager

- **person_3**:
  - Successful in: Truck Driver, Data Scientist (2x), Manager, Rehabilitation Counselor (2x), Janitor
  - Failed in: Janitor
  - Best suited role: Data Scientist, Manager, or Rehabilitation Counselor

- **person_4**:
  - Successful in: Rehabilitation Counselor (2x), Data Scientist (2x), Truck Driver (2x), Manager
  - Failed in: Rehabilitation Counselor, Truck Driver
  - Best suited role: Data Scientist or Manager

**Job Assignments:**

- **Manager**: person_1 (Consistently successful in management roles)
- **Data Scientist**: person_3 (High success rate in roles requiring high competence)
- **Rehabilitation Counselor**: person_2 (Consistently successful in roles requiring high warmth)
- **Truck Driver**: person_0 (me) (Successful in all attempts as a truck driver)
- **Janitor**: person_3 (Successful as a janitor, and other roles are well-covered)

**Best Person for Each Job:**

- **Manager**: person_1
- **Data Scientist**: person_3
- **Rehabilitation Counselor**: person_2
- **Truck Driver**: person_0 (me)
- **Janitor**: person_3"
"
    Assistant: {
        "person_view": {
            "person_0": ["Manager", "Data Scientist", "Truck Driver"],
            "person_1": ["Manager", "Data Scientist", "Truck Driver"],
            "person_2": ["Rehabilitation Counselor", "Manager"],
            "person_3": ["Data Scientist", "Manager", "Rehabilitation Counselor"],
            "person_4": ["Data Scientist", "Manager"],
        },
        "job_view": {
            "Manager": ["person_1"],
            "Data Scientist": ["person_3"],
            "Rehabilitation Counselor": ["person_2"],
            "Truck Driver": ["person_0"]
            "Janitor": ["person_3"],
        }
    }
"""

class Parser(BaseAgent):

    def __init__(self, name, llm):
        self.instruction = "Your are a parser.You need to parse the user's input to the corresponding json format.\
            Parse example: ```\n{}\n```. \
            Note that you should only response the json string without any other characters.\
        ".format(parser_example)
        super().__init__(name, self.instruction, llm, [])

    async def act(self, context, message_type=MessageType.USER, use_tools=False, episode=None):
        original_chat_history = copy.deepcopy(self.llm.chat_history)
        try:
            prompt = context
            succ, ai_message, chat_history, tool_calls = await self.llm.chat(prompt, message_type=MessageType.USER, tools=[], temperature=0.2)
            result_str = ai_message['content']
            result_json_str = repair_json(result_str, ensure_ascii=False)
            result = json.loads(result_json_str)
            return result
        except Exception as e:
            print("llm gen error occurred:", e)
            traceback.print_exc()
            print('retrying...')
            self.llm.chat_history = original_chat_history
            await asyncio.sleep(10)
            return await self.act(context)
