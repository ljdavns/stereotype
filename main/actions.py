from agent.player import Player
from util.excel_logger import ExcelLogger
from typing import List, Dict

class ActionManager:
    def __init__(self, message_queue: List[Dict], logger: ExcelLogger):
        self.message_queue = message_queue
        self.logger = logger

    async def talk_to_someone(self, episode: int, agent: Player, target_name: str, message: str):
        player_name = agent.name
        self.message_queue.append({
            'episode': episode,
            'action': 'talk_to_someone',
            'source': player_name,
            'target': [target_name],
            'message': message
        })
        self.logger.log_game_record(self.message_queue[-1])
        print(self.message_queue[-1])

    async def talk_to_some_people(self, episode: int, agent: Player, target_name_list: list[str], message: str):
        player_name = agent.name
        self.message_queue.append({
            'episode': episode,
            'action': 'talk_to_some_people',
            'source': player_name,
            'target': target_name_list,
            'message': message
        })
        self.logger.log_game_record(self.message_queue[-1])
        print(self.message_queue[-1])

    async def talk_to_public(self, episode: int, agent: Player, message: str):
        player_name = agent.name
        self.message_queue.append({
            'episode': episode,
            'action': 'talk_to_public',
            'source': player_name,
            'target': 'all',
            'message': message
        })
        self.logger.log_game_record(self.message_queue[-1])
        print(self.message_queue[-1])

    async def study(self, episode: int, agent: Player, job_type: str):
        player_name = agent.name
        self.message_queue.append({
            'episode': episode,
            'action': 'study',
            'source': player_name,
            'target': [player_name],
            'message': f"I have studied {job_type}"
        })
        self.logger.log_game_record(self.message_queue[-1])
        print(self.message_queue[-1])

    async def reflection(self, episode: int, agent: Player, reflection_topic: str):
        player_name = agent.name
        reflection_result = await agent.act(f"I need to reflect on {reflection_topic}")
        self.message_queue.append({
            'episode': episode,
            'action': 'reflection',
            'source': player_name,
            'target': ['none'],
            'message': f"{player_name} has reflected on {reflection_topic} and the result is {reflection_result}"
        })
        self.logger.log_game_record(self.message_queue[-1])
        print(self.message_queue[-1])

    @property
    def function_map(self):
        return {
            'talk_to_someone': self.talk_to_someone,
            'talk_to_some_people': self.talk_to_some_people,
            'talk_to_public': self.talk_to_public,
            # 'study': self.study,
            # 'reflection': self.reflection
        }