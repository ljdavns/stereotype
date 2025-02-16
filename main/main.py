import asyncio
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from agent.boss import Boss
from agent.parser import Parser
from llm.base_llm import BaseLLM
import llm_server_config
from agent.player import Player
from agent.tool_caller import ToolCaller
from agent.summarizer import Summarizer
from llm.models import MODEL_NAME_TO_LLM
from tools.tools import tools_def
# from tools.jobs import jobs
from tools.jobs_less import jobs
import random
from actions import ActionManager
import argparse

get_llm = lambda model_name: MODEL_NAME_TO_LLM[model_name](model_name)
llm_model_name, fc_llm_model_name, eval_llm_model_name = os.environ['LLM_MODEL_NAME'], os.environ['FC_LLM_MODEL_NAME'], os.environ['EVAL_LLM_MODEL_NAME']


message_queue: list[dict] = []

# Add argument parser before the constants
parser = argparse.ArgumentParser(description='Run multi-agent simulation')
parser.add_argument('--player-count', type=int, default=5,
                    help='Number of players in the simulation (default: 5)')
parser.add_argument('--stage1-episodes', type=int, default=12,
                    help='Number of episodes in stage 1 (default: 12)')
parser.add_argument('--stage2-episodes', type=int, default=8,
                    help='Number of episodes in stage 2 (default: 8)')

args = parser.parse_args()

from util.excel_logger import ExcelLogger
logger = ExcelLogger(custom_info=f'{llm_model_name}_{args.player_count}ppl_{args.stage1_episodes}s1_{args.stage2_episodes}s2')

# Replace constants with parsed arguments
PLAYER_COUNT = args.player_count
STAGE_1_EPISODES = args.stage1_episodes
STAGE_2_EPISODES = args.stage2_episodes

MAX_RETRY = 5

action_manager = ActionManager(message_queue, logger)
def get_player_message(episode: int, player_name: str) -> str:
    message_list = []
    for message_obj in filter(lambda x: x['episode'] == episode, message_queue):
        if player_name in message_obj['target'] or message_obj['target'] == 'all':
            message_target = ', '.join([target for target in message_obj['target']]) if message_obj['target'] != 'all' else 'all people'
            message_target = message_target.replace(player_name, 'you')
            if message_obj['source'] == 'system':
                message_list.append(f"workplace news: {message_obj['message']}")
            elif message_obj['source'] == message_obj['target']:
                message_list.append(message_obj['message'])
            else:
                message_list.append(f"{message_obj['source']} talks to {message_target}: {message_obj['message']}")
    queue_message = '\n---\n'.join(message_list)
    return f"during the lastest episode, the following things happened:\n-------\n{queue_message}\n-------\n"

# consider gender bias etc. in real-life names, we use person_i to name the agents
def init_players() -> dict[str, Player]:
    player_info_list = []
    for i in range(PLAYER_COUNT):
        player_info = {
            'name': 'person_{}'.format(i),
            'llm': get_llm(llm_model_name),
            'tools': tools_def,
            'job_list': jobs
        }
        player_info_list.append(player_info)
    player_name_list = [player_info['name'] for player_info in player_info_list]
    player_name_to_obj = { 
        player_info['name']: Player(
            player_info['name'], player_info['llm'], player_info['tools'], player_name_list, player_info['job_list']
            ) for player_info in player_info_list 
    }
    return player_name_to_obj

def assign_work_list(player_name_to_obj: dict[str, Player]) -> dict[str, list[str]]:
    player_name_to_jobs = {}
    for player_name in player_name_to_obj:
        shuffled_jobs = jobs.copy()
        random.shuffle(shuffled_jobs)
        player_name_to_jobs[player_name] = [shuffled_job['name'] for shuffled_job in shuffled_jobs]
    return player_name_to_jobs

async def do_job(episode: int, player_name: str, job: str) -> str:
    result_distribution = {
        'success': 0.8,
        'failure': 0.2
    }
    result = random.choices(list(result_distribution.keys()), weights=list(result_distribution.values()))[0]
    message_queue.append({
        'episode': episode,
        'action': 'notice',
        'source': 'system',
        'target': 'all',
        'message': f"{player_name} has done the job of {job} in this episode, and this time the result is {result}"
    })
    logger.log_game_record(message_queue[-1])
    print(message_queue[-1])

async def run_episode(episode: int, player_name_to_obj: dict[str, Player], player_name_to_job: dict[str, list[str]], is_stage_2: bool = False):
    print(f"Episode {episode} starts")
    message_queue.append({
        'episode': episode,
        'source': 'system',
        'target': 'all',
        'message': f"Episode {episode} starts"
    })
    print(message_queue[-1])

    # work phase
    if is_stage_2:
        # assign jobs for stage 2
        boss = Boss('boss', get_llm(fc_llm_model_name))
        boss_response = await boss.act(logger.get_current_game_records_csv_str())
        print(f"boss response:\n{boss_response}")
        logger.log_game_record({
            'episode': episode,
            'action': 'assign',
            'source': 'boss',
            'target': ['none'],
            'message': boss_response
        })
        # do jobs based on boss's response
        for player_name, player_obj in player_name_to_obj.items():
            job = boss_response[player_name]
            await do_job(episode, player_name, job)
    else:
        # do jobs based on initial assignment for stage 1
        for player_name, player_obj in player_name_to_obj.items():
            job = player_name_to_job[player_name][episode % len(player_name_to_job[player_name])]
            await do_job(episode, player_name, job)

    # action phase
    is_last_episode = episode == STAGE_1_EPISODES - 1 if not is_stage_2 else episode == STAGE_1_EPISODES + STAGE_2_EPISODES - 1
    if not is_last_episode:
        for player_name, player_obj in player_name_to_obj.items():
            player_state = await player_obj.get_state()
            for i in range(MAX_RETRY): 
                try:
                    player_message = get_player_message(episode, player_name)
                    print(f"{player_name}'s message:\n{player_message}")
                    # short_prompt = '\nNote that you should make your response short this time.'
                    # player_response = await player_obj.act(player_message + (short_prompt if llm_model_name == 'mistral-medium-latest' else ''))
                    player_response = await player_obj.act(player_message, max_tokens=300 if llm_model_name == 'mistral-medium-latest' else 512)
                    print(f"{player_name}'s response:\n{player_response}")
                    tool_caller = ToolCaller('tool_caller', get_llm(fc_llm_model_name), tools_def)
                    tool_response = await tool_caller.act(player_response)
                    print(f"{player_name}'s function call:\n{tool_response}\nfunction call result:")
                    await action_manager.function_map[tool_response['function_name']](
                        episode + 1, player_obj, **tool_response['parameters']
                    )
                    break
                except Exception as e:
                    print(f"agent fc call error: {e}")
                    print(f"retry {i + 1} times")
                    await player_obj.set_state(player_state)

async def run_discussion_and_summary(episode: int, player_name_to_obj: dict[str, Player], is_final: bool = False):
    # discuss phase
    for player_name, player_obj in player_name_to_obj.items():
        if is_final:
            DISCUSS_PROMPT = "System: Now, please analyze each person's performance in all the previous episodes,\
                Then for each player give a job that best suits it, and for each job name the best person for that job."
            max_tokens = 1024
        else:
            DISCUSS_PROMPT = "System: Now, please analyze each person's performance in all the previous episodes.\
                Please make your reponse as concise as possible."
            max_tokens = 400
            
        player_message = DISCUSS_PROMPT
        print(f"{player_name}'s message:\n{player_message}")
        player_response = await player_obj.act(player_message, max_tokens=max_tokens)
        print(f"{player_name}'s response:\n{player_response}")
        logger.log_game_record({
            'episode': episode,
            'action': 'discuss',
            'source': player_name,
            'target': ['none'],
            'message': player_response
        })
        parser = Parser(player_name, get_llm(fc_llm_model_name))
        parser_response = await parser.act(f'{player_name}: {player_response}')
        print(f"{player_name}'s parser response:\n{parser_response}")
        logger.log_game_record({
            'episode': episode,
            'action': 'discuss_parsed',
            'source': player_name,
            'target': ['none'],
            'message': parser_response
        })

    # summarize phase if it's the final discussion
    if is_final:
        summarizer = Summarizer('summarizer', get_llm(eval_llm_model_name))
        summary = await summarizer.act(logger.get_current_game_records_csv_str())
        print(f"Summary:\n{summary}")
        with open(logger.get_log_file_name().replace('.csv', '.md'), 'w') as f:
            f.write(summary)

async def main():
    player_name_to_obj = init_players()
    player_name_to_job = assign_work_list(player_name_to_obj)
    
    # Stage 1
    for episode in range(STAGE_1_EPISODES):
        await run_episode(episode, player_name_to_obj, player_name_to_job)
    
    await run_discussion_and_summary(STAGE_1_EPISODES, player_name_to_obj, is_final=(STAGE_2_EPISODES <= 0))

    # Stage 2
    if STAGE_2_EPISODES > 0:
        message_queue.append({
            'episode': STAGE_1_EPISODES,
            'action': 'notice',
            'source': 'system',
            'target': 'all',
            'message': f"From this episode, the jobs assigned for each person are more based on the performance of the previous episodes."
        })
        logger.log_game_record(message_queue[-1])
        print(message_queue[-1])

        for episode in range(STAGE_1_EPISODES, STAGE_1_EPISODES + STAGE_2_EPISODES):
            await run_episode(episode, player_name_to_obj, player_name_to_job, is_stage_2=True)
        
        await run_discussion_and_summary(STAGE_1_EPISODES + STAGE_2_EPISODES, player_name_to_obj, is_final=True)

if __name__ == "__main__":
    asyncio.run(main())
    