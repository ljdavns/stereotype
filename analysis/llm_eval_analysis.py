import json
from pathlib import Path
import pandas as pd

def analyze_model_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Define total episode thresholds to analyze
    total_episodes = [10, 14, 16, 20, 25]
    
    results = {
        'with_boss': {},
        'without_boss': {},
        'overall': {}  # New category for all cases
    }
    
    # Analyze all cases
    for total_ep in total_episodes:
        df_filtered = df[df['stage1_episodes'] + df['stage2_episodes'] == total_ep]
        
        if len(df_filtered) == 0:
            continue
            
        metrics = {
            'have_stereotype': round(float(df_filtered['have_stereotype'].mean()), 2),
            'have_strong_stereotype': round(float(df_filtered['have_strong_stereotype'].mean()), 2),
            'halo_effect': round(float(df_filtered['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
            'confirmation_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
            'role_congruity': round(float(df_filtered['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
            'self_serving_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
        }
        
        results['overall'][total_ep] = metrics
    
    # Add metrics for all episodes combined
    metrics = {
        'have_stereotype': round(float(df['have_stereotype'].mean()), 2),
        'have_strong_stereotype': round(float(df['have_strong_stereotype'].mean()), 2),
        'halo_effect': round(float(df['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
        'confirmation_bias': round(float(df['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
        'role_congruity': round(float(df['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
        'self_serving_bias': round(float(df['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
    }
    results['overall']['all_episodes'] = metrics
    
    # Analyze cases without boss (stage2 == 0)
    df_without_boss = df[df['stage2_episodes'] == 0]
    for total_ep in total_episodes:
        df_filtered = df_without_boss[df_without_boss['stage1_episodes'] == total_ep]
        
        if len(df_filtered) == 0:
            continue
            
        metrics = {
            'have_stereotype': round(float(df_filtered['have_stereotype'].mean()), 2),
            'have_strong_stereotype': round(float(df_filtered['have_strong_stereotype'].mean()), 2),
            'halo_effect': round(float(df_filtered['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
            'confirmation_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
            'role_congruity': round(float(df_filtered['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
            'self_serving_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
        }
        
        results['without_boss'][total_ep] = metrics
    
    # Add metrics for all episodes without boss
    metrics = {
        'have_stereotype': round(float(df_without_boss['have_stereotype'].mean()), 2),
        'have_strong_stereotype': round(float(df_without_boss['have_strong_stereotype'].mean()), 2),
        'halo_effect': round(float(df_without_boss['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
        'confirmation_bias': round(float(df_without_boss['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
        'role_congruity': round(float(df_without_boss['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
        'self_serving_bias': round(float(df_without_boss['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
    }
    results['without_boss']['all_episodes'] = metrics
    
    # Analyze cases with boss (stage2 != 0)
    df_with_boss = df[df['stage2_episodes'] != 0]
    for total_ep in total_episodes:
        df_filtered = df_with_boss[df_with_boss['stage1_episodes'] + df_with_boss['stage2_episodes'] == total_ep]
        
        if len(df_filtered) == 0:
            continue
            
        metrics = {
            'have_stereotype': round(float(df_filtered['have_stereotype'].mean()), 2),
            'have_strong_stereotype': round(float(df_filtered['have_strong_stereotype'].mean()), 2),
            'halo_effect': round(float(df_filtered['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
            'confirmation_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
            'role_congruity': round(float(df_filtered['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
            'self_serving_bias': round(float(df_filtered['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
        }
        
        results['with_boss'][total_ep] = metrics
    
    # Add metrics for all episodes with boss
    metrics = {
        'have_stereotype': round(float(df_with_boss['have_stereotype'].mean()), 2),
        'have_strong_stereotype': round(float(df_with_boss['have_strong_stereotype'].mean()), 2),
        'halo_effect': round(float(df_with_boss['social_dynamics'].apply(lambda x: x['halo_effect']).mean()), 2),
        'confirmation_bias': round(float(df_with_boss['social_dynamics'].apply(lambda x: x['confirmation_bias']).mean()), 2),
        'role_congruity': round(float(df_with_boss['social_dynamics'].apply(lambda x: x['role_congruity']).mean()), 2),
        'self_serving_bias': round(float(df_with_boss['social_dynamics'].apply(lambda x: x['self_serving_bias']).mean()), 2)
    }
    results['with_boss']['all_episodes'] = metrics
        
    return results

def main():
    # Get all json files
    json_files = list(Path('./llm_eval').glob('extracted_reports_*.json'))
    
    all_results = {}
    for json_file in json_files:
        model_name = json_file.stem.split('extracted_reports_')[1]
        results = analyze_model_results(json_file)
        all_results[model_name] = results
    
    json.dump(all_results, open('./llm_eval/llm_eval_analysis_all.json', 'w'), indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
