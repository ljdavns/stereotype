import asyncio
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from agent.report_extractor import ReportExtractor
from llm.models import MODEL_NAME_TO_LLM
from tools.tools import tools_def
# from tools.jobs import jobs
from tools.jobs_less import jobs
import random
from tqdm import tqdm


get_llm = lambda model_name: MODEL_NAME_TO_LLM[model_name](model_name)
# fc_llm_model_name = os.environ['FC_LLM_MODEL_NAME']
# fc_llm_model_name = 'gemini-2.0-flash-exp'
llm_model_name = os.environ['LLM_MODEL_NAME']

async def main():
    # Get all log files from ./logs directory
    log_dir = Path('../logs')
    report_files = list(log_dir.glob('*.md'))  
    log_files = list(log_dir.glob('*.csv'))
    
    results = []
    
    # Process each log file
    for log_file, report_file in tqdm(zip(log_files, report_files)):
        with open(log_file, 'r', encoding='utf-8') as f, open(report_file, 'r', encoding='utf-8') as f2:
            log_content = f.read()
            report_content = f2.read()
            content = log_content + '\n' + report_content
            extractor = ReportExtractor('report_extractor', get_llm(llm_model_name))
            report_json = await extractor.act(content)
            ppl = int(log_file.name.split('ppl_')[0].split('_')[-1])
            s1 = int(log_file.name.split('s1_')[0].split('_')[-1])
            s2 = int(log_file.name.split('s2_')[0].split('_')[-1])
            if report_json:
                results.append({
                    'file_name': log_file.name,
                    'agent_count': ppl,
                    'stage1_episodes': s1,
                    'stage2_episodes': s2,
                    **report_json
                })
                print(results[-1])
    os.makedirs('./llm_eval', exist_ok=True)
    # Save results to JSON file
    output_file = f'./llm_eval/extracted_reports_{llm_model_name}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    asyncio.run(main())
