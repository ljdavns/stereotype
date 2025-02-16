import pandas as pd
import os
from datetime import datetime
import json

python_file_path = os.path.abspath(__file__)
LOG_FILE_DIR = os.path.join(os.path.dirname(python_file_path), '../logs')
if not os.path.exists(LOG_FILE_DIR):
    os.makedirs(LOG_FILE_DIR)

class ExcelLogger:
    def __init__(self, custom_info=''):
        self.game_record_list = []
        # Add timestamp to filename to avoid duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f'{LOG_FILE_DIR}/stereotype_multiagent_{custom_info}_{timestamp}.csv'
        
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=['episode', 'action', 'source', 'target', 'message']).to_csv(self.csv_file, index=False)
            
    def get_log_file_name(self):
        return self.csv_file

    def log_game_record(self, record):
        self.game_record_list.append(record)
        df = pd.DataFrame([record])
        df.to_csv(self.csv_file, mode='a', header=False, index=False)
    
    def get_current_game_records_str(self):
        return json.dumps(self.game_record_list, indent=4)

    def get_current_game_records_csv_str(self):
        return pd.DataFrame(self.game_record_list).to_csv(index=False)
    