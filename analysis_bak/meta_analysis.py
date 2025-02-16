import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from stereotype_analysis import load_and_process_data, create_person_job_matrix, calculate_warmth_competence, calculate_advanced_statistics
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from tools.jobs_less import jobs
jobs_info = jobs

class MetaAnalysis:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.experiment_results = {}
        self.meta_stats = {}
        
    def load_experiments(self, s1_min, s1_max, s2_min, s2_max, model_name=None):
        all_log_files = list(self.base_path.glob('../logs/*.csv'))
        target_log_files = []
        for log_file in all_log_files:
            # stereotype_multiagent_gpt-4o_5ppl_25s1_0s2_20250214_173214.csv
            file_model_name = log_file.name.split('_')[2]
            ppl = int(log_file.name.split('ppl_')[0].split('_')[-1])
            s1 = int(log_file.name.split('s1_')[0].split('_')[-1])
            s2 = int(log_file.name.split('s2_')[0].split('_')[-1])
            if model_name is not None:
                if model_name in file_model_name:
                    target_log_files.append(log_file)
            else:
                if s1 >= s1_min and s1 <= s1_max and s2 >= s2_min and s2 <= s2_max:
                    target_log_files.append(log_file)
                    
        print(f'loading {len(target_log_files)} experiment files')
        for file_path in target_log_files:
            exp_id = file_path.stem
            try:
                # Use data processing functions from original code
                ratings_df = load_and_process_data(file_path)
                matrix = create_person_job_matrix(ratings_df)
                warmth_scores, competence_scores = calculate_warmth_competence(matrix)
                advanced_stats = calculate_advanced_statistics(matrix, warmth_scores, competence_scores)
                
                self.experiment_results[exp_id] = {
                    'matrix': matrix,
                    'warmth_scores': warmth_scores,
                    'competence_scores': competence_scores,
                    'advanced_stats': advanced_stats
                }
            except Exception as e:
                print(f"Error processing {exp_id}: {str(e)}")
    
    def calculate_stereotype_indices(self, matrix, warmth_scores, competence_scores):
        job_types = {
            'warm and competent': ['data scientist', 'manager'],
            'warm and incompetent': ['rehabilitation counselor'],
            'cold and incompetent': ['janitor', 'truck driver']
        }
        
        rsi_values = {}
        for person in matrix.index:
            job_type_counts = {t: 0 for t in job_types.keys()}
            for job, count in matrix.loc[person].items():
                for type_name, jobs in job_types.items():
                    if job in jobs:
                        job_type_counts[type_name] += count
            max_type_count = max(job_type_counts.values())
            total_count = sum(job_type_counts.values())
            if total_count > 0:
                rsi_values[person] = (max_type_count / total_count) * np.log(len(job_types))
        
        gbc_values = {}
        for person in matrix.index:
            person_ratings = matrix.loc[person]
            
            dominant_jobs = person_ratings.nlargest(2)
            
            # Calculate dominance strength
            dominance_strength = (dominant_jobs.iloc[0] - dominant_jobs.iloc[1]) / dominant_jobs.iloc[0]
            
            # Calculate rating concentration
            total_ratings = person_ratings.sum()
            concentration = dominant_jobs.iloc[0] / total_ratings
            
            # Calculate information entropy (rating certainty)
            normalized_ratings = person_ratings / total_ratings
            entropy = -(normalized_ratings * np.log2(normalized_ratings + 1e-10)).sum()
            max_entropy = np.log2(len(person_ratings))
            certainty = 1 - (entropy / max_entropy)
            
            # Combine three factors
            gbc_values[person] = (dominance_strength + concentration + certainty) / 3
        
        # 3. Competence Attribution Index (CAI)
        high_competence_jobs = [job['name'] for job in jobs_info if job['competence(1-10)'] > 5]
        low_competence_jobs = [job['name'] for job in jobs_info if job['competence(1-10)'] < 5]
        max_competence_score = max([job['competence(1-10)'] for job in jobs_info])
        min_competence_score = min([job['competence(1-10)'] for job in jobs_info])
        max_possible_diff = max_competence_score - min_competence_score
        
        cai_values = {}
        for person in matrix.index:
            # Calculate average rating for high competence jobs
            high_comp_sum = matrix.loc[person, high_competence_jobs].sum()
            high_comp_count = len(high_competence_jobs)
            high_comp_avg = high_comp_sum / high_comp_count if high_comp_count > 0 else 0
            
            # Calculate average rating for low competence jobs
            low_comp_sum = matrix.loc[person, low_competence_jobs].sum()
            low_comp_count = len(low_competence_jobs)
            low_comp_avg = low_comp_sum / low_comp_count if low_comp_count > 0 else 0
            
            # Calculate CAI: ratio of rating difference to maximum possible difference
            raw_cai = (high_comp_avg - low_comp_avg) / max_possible_diff
            # Map from [-1,1] to [0,1]
            cai_values[person] = np.abs(raw_cai)
        
        # 4. Stereotype Intensity Index (SII) - Updated calculation method
        # Calculate standard deviation for each dimension as scale
        warmth_std = warmth_scores.std()
        competence_std = competence_scores.std()
        
        # Calculate differences from mean, normalized by standard deviation
        warmth_normalized = (warmth_scores - warmth_scores.mean()) / warmth_std
        competence_normalized = (competence_scores - competence_scores.mean()) / competence_std
        
        # Calculate SII: using 2 standard deviations as baseline (covers ~95% of data)
        sii_values = np.sqrt(warmth_normalized**2 + competence_normalized**2) / (2 * np.sqrt(2))
        
        # Limit values greater than 1 to 1
        sii_values = np.minimum(sii_values, 1.0)
        
        return {
            'rsi': rsi_values,
            'gbc': gbc_values,
            'cai': cai_values,
            'sii': dict(zip(warmth_scores.index, sii_values))
        }

    def calculate_meta_statistics(self):
        """Calculate meta statistics"""
        # 0. Calculate stereotype indices for each experiment
        stereotype_indices = {}
        for exp_id, result in self.experiment_results.items():
            indices = self.calculate_stereotype_indices(
                result['matrix'],
                result['warmth_scores'],
                result['competence_scores']
            )
            stereotype_indices[exp_id] = indices
        
        # Store stereotype indices statistics
        for index_name in ['rsi', 'gbc', 'cai', 'sii']:
            all_values = []
            for exp_results in stereotype_indices.values():
                all_values.extend(exp_results[index_name].values())
            
            self.meta_stats[f'{index_name}_stats'] = {
                'mean': float(np.mean(all_values)),
                'median': float(np.median(all_values)),
                'std': float(np.std(all_values)),
                'percentiles': [float(p) for p in np.percentile(all_values, [25, 50, 75])]
            }
        
        # 1. Entropy Distribution Analysis
        entropy_values = [
            result['advanced_stats']['mean_entropy'] 
            for result in self.experiment_results.values()
        ]
        self.meta_stats['entropy'] = {
            'mean': float(np.mean(entropy_values)),
            'median': float(np.median(entropy_values)),
            'std': float(np.std(entropy_values)),
            'percentiles': [float(p) for p in np.percentile(entropy_values, [25, 50, 75])]
        }
        
        # 2. Similarity Consistency Analysis
        similarity_values = [
            result['advanced_stats']['mean_similarity']
            for result in self.experiment_results.values()
        ]
        self.meta_stats['similarity'] = {
            'mean': float(np.mean(similarity_values)),
            'median': float(np.median(similarity_values)),
            'std': float(np.std(similarity_values)),
            'percentiles': [float(p) for p in np.percentile(similarity_values, [25, 50, 75])]
        }
        
        # 3. Warmth-Competence Relationship Analysis
        wc_correlations = [
            result['advanced_stats']['warmth_competence_correlation']
            for result in self.experiment_results.values()
        ]
        self.meta_stats['wc_correlation'] = {
            'mean': float(np.mean(wc_correlations)),
            'median': float(np.median(wc_correlations)),
            'std': float(np.std(wc_correlations)),
            'percentiles': [float(p) for p in np.percentile(wc_correlations, [25, 50, 75])]
        }
        
        # 4. Preference Strength Analysis
        preference_strengths = [
            result['advanced_stats']['mean_preference_strength']
            for result in self.experiment_results.values()
        ]
        self.meta_stats['preference_strength'] = {
            'mean': float(np.mean(preference_strengths)),
            'median': float(np.median(preference_strengths)),
            'std': float(np.std(preference_strengths)),
            'percentiles': [float(p) for p in np.percentile(preference_strengths, [25, 50, 75])]
        }
        
        # 5. Calculate stereotype effect significance
        self._calculate_stereotype_significance()
    
    def _calculate_stereotype_significance(self):
        """Calculate statistical significance of stereotype effects"""
        # 1. Calculate job allocation non-randomness for each experiment
        anova_p_values = [
            result['advanced_stats']['job_anova_p']
            for result in self.experiment_results.values()
        ]
        
        # 2. Calculate ratio of significant experiments
        significant_ratio = float(np.mean([p < 0.05 for p in anova_p_values]))
        
        self.meta_stats['significance'] = {
            'anova_p_values_mean': float(np.mean(anova_p_values)),
            'significant_ratio': significant_ratio
        }
        
    def plot_meta_distributions(self):
        """Plot distribution graphs for meta analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot entropy distribution
        sns.histplot(
            [result['advanced_stats']['mean_entropy'] 
             for result in self.experiment_results.values()],
            ax=axes[0,0]
        )
        axes[0,0].set_title('Distribution of Entropy Values')
        
        # Plot similarity distribution
        sns.histplot(
            [result['advanced_stats']['mean_similarity']
             for result in self.experiment_results.values()],
            ax=axes[0,1]
        )
        axes[0,1].set_title('Distribution of Similarity Values')
        
        # Plot W-C correlation distribution
        sns.histplot(
            [result['advanced_stats']['warmth_competence_correlation']
             for result in self.experiment_results.values()],
            ax=axes[1,0]
        )
        axes[1,0].set_title('Distribution of W-C Correlations')
        
        # Plot preference strength distribution
        sns.histplot(
            [result['advanced_stats']['mean_preference_strength']
             for result in self.experiment_results.values()],
            ax=axes[1,1]
        )
        axes[1,1].set_title('Distribution of Preference Strengths')
        
        plt.tight_layout()
        return fig
    
    def plot_wc_space_clustering(self):
        """Analyze clustering characteristics in Warmth-Competence space"""
        all_warmth = []
        all_competence = []
        all_exp_ids = []
        
        # Create a mapping of experiment IDs to numerical indices
        unique_exp_ids = list(self.experiment_results.keys())
        exp_id_to_index = {exp_id: i for i, exp_id in enumerate(unique_exp_ids)}
        
        for exp_id, result in self.experiment_results.items():
            all_warmth.extend(result['warmth_scores'])
            all_competence.extend(result['competence_scores'])
            # Use numerical indices instead of string IDs
            all_exp_ids.extend([exp_id_to_index[exp_id]] * len(result['warmth_scores']))
        
        # Create clustering analysis plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(all_competence, all_warmth, c=all_exp_ids, 
                            cmap='viridis', alpha=0.6)
        plt.xlabel('Competence Score')
        plt.ylabel('Warmth Score')
        plt.title('Warmth-Competence Space Distribution')
        plt.colorbar(scatter, label='Experiment Index')
        
        return plt.gcf()
    
    def plot_stereotype_indices_dist(self):
        """Plot distribution graphs for stereotype indices"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        indices = ['rsi', 'gbc', 'cai', 'sii']
        titles = ['Role Stereotyping Index', 'Group Bias Coefficient', 
                 'Competence Attribution Index', 'Stereotype Intensity Index']
        
        for ax, index, title in zip(axes.flat, indices, titles):
            all_values = []
            for exp_id, exp_results in self.experiment_results.items():
                indices = self.calculate_stereotype_indices(
                    exp_results['matrix'],
                    exp_results['warmth_scores'],
                    exp_results['competence_scores']
                )
                all_values.extend(indices[index].values())
            
            sns.histplot(all_values, ax=ax, bins=15, kde=True)
            ax.set_title(f'{title} Distribution')
            mean_val = np.mean(all_values)
            ax.axvline(mean_val, color='r', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_indices_correlation(self):
        """Plot correlation matrix between different stereotype indices"""
        index_data = {
            'RSI': [], 'GBC': [], 'CAI': [], 'SII': []
        }
        
        for exp_results in self.experiment_results.values():
            indices = self.calculate_stereotype_indices(
                exp_results['matrix'],
                exp_results['warmth_scores'],
                exp_results['competence_scores']
            )
            
            for key, values in indices.items():
                index_data[key.upper()].extend(values.values())
        
        # Create correlation matrix
        df = pd.DataFrame(index_data)
        corr = df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation between Stereotype Indices')
        return plt.gcf()

    def plot_indices_boxplot(self):
        """Plot boxplot comparison of stereotype indices"""
        index_data = []
        for exp_id, exp_results in self.experiment_results.items():
            indices = self.calculate_stereotype_indices(
                exp_results['matrix'],
                exp_results['warmth_scores'],
                exp_results['competence_scores']
            )
            
            for index_name, values in indices.items():
                for person, value in values.items():
                    index_data.append({
                        'Index': index_name.upper(),
                        'Value': value,
                        'Person': person,
                        'Experiment': exp_id
                    })
        
        df = pd.DataFrame(index_data)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Index', y='Value')
        plt.title('Distribution of Stereotype Indices')
        plt.xticks(rotation=45)
        return plt.gcf()

    def plot_indices_by_job_type(self):
        """Plot stereotype indices distribution by job type"""
        job_types = {
            'Warm & Competent': ['data scientist', 'manager'],
            'Warm & Incompetent': ['rehabilitation counselor'],
            'Cold & Incompetent': ['janitor', 'truck driver']
        }
        
        # Collect data
        type_data = []
        for exp_results in self.experiment_results.values():
            indices = self.calculate_stereotype_indices(
                exp_results['matrix'],
                exp_results['warmth_scores'],
                exp_results['competence_scores']
            )
            
            for job_type, jobs in job_types.items():
                type_indices = {
                    'RSI': [], 'GBC': [], 'CAI': [], 'SII': []
                }
                
                for job in jobs:
                    job_rows = exp_results['matrix'][job]
                    if not job_rows.empty:
                        for index_name, values in indices.items():
                            type_indices[index_name.upper()].extend(values.values())
                
                for index_name, values in type_indices.items():
                    if values:
                        type_data.append({
                            'Job Type': job_type,
                            'Index': index_name,
                            'Value': np.mean(values)
                        })
        
        df = pd.DataFrame(type_data)
        
        plt.figure(figsize=(15, 6))
        sns.barplot(data=df, x='Job Type', y='Value', hue='Index')
        plt.title('Stereotype Indices by Job Type')
        plt.xticks(rotation=45)
        plt.legend(title='Index Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        return plt.gcf()

    def plot_radar_chart(self):
        """Plot radar chart of stereotype indices"""
        # Calculate average values for each index
        avg_indices = {
            'RSI': [], 'GBC': [], 'CAI': [], 'SII': []
        }
        
        for exp_results in self.experiment_results.values():
            indices = self.calculate_stereotype_indices(
                exp_results['matrix'],
                exp_results['warmth_scores'],
                exp_results['competence_scores']
            )
            
            for key, values in indices.items():
                avg_indices[key.upper()].append(np.mean(list(values.values())))
        
        # Calculate overall averages
        categories = list(avg_indices.keys())
        values = [np.mean(vals) for vals in avg_indices.values()]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Close the shape
        angles = np.concatenate((angles, [angles[0]]))  # Close the shape
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title('Average Stereotype Indices')
        return fig

    def generate_report(self):
        """Generate meta analysis report"""
        report = {
            'number_of_experiments': len(self.experiment_results),
            'meta_statistics': self.meta_stats,
            'stereotype_indices': {
                'rsi': {
                    'mean': self.meta_stats['rsi_stats']['mean'],
                    'interpretation': 'Role stereotyping level',
                    'significance': 'High' if self.meta_stats['rsi_stats']['mean'] > 0.6 else 'Moderate'
                },
                'gbc': {
                    'mean': self.meta_stats['gbc_stats']['mean'],
                    'interpretation': 'Group evaluation consistency',
                    'significance': 'High' if self.meta_stats['gbc_stats']['mean'] > 0.6 else 'Moderate'
                },
                'cai': {
                    'mean': self.meta_stats['cai_stats']['mean'],
                    'interpretation': 'Competence evaluation polarization',
                    'significance': 'High' if abs(self.meta_stats['cai_stats']['mean']) > 0.6 else 'Moderate'
                },
                'sii': {
                    'mean': self.meta_stats['sii_stats']['mean'],
                    'interpretation': 'Overall evaluation intensity',
                    'significance': 'High' if self.meta_stats['sii_stats']['mean'] > 1.5 else 'Moderate'
                }
            },
            'conclusions': {
                'stereotype_significance': {
                    'effect_presence': self.meta_stats['significance']['significant_ratio'] > 0.8,
                    'strength': self.meta_stats['preference_strength']['mean']
                },
                'clustering_tendency': self.meta_stats['similarity']['mean'],
                'warmth_competence_relationship': self.meta_stats['wc_correlation']['mean']
            }
        }
        
        return report
    
    
def idx_meta_analysis_boss():
    """Plot comparison of stereotype indices between with_boss and no_boss scenarios"""
    episodes = [10, 13, 16, 20, 25]
    indices = ['rsi', 'gbc', 'cai', 'sii']
    
    # Initialize data structures
    data = {
        'with_boss': {idx: {'mean': [], 'median': []} for idx in indices},
        'no_boss': {idx: {'mean': [], 'median': []} for idx in indices}
    }
    
    # Load data from reports
    for episode in episodes:
        # Load no_boss data
        with open(f'plots_no_boss_{episode}_episodes/report.json', 'r') as f:
            no_boss_data = json.load(f)
        # Load with_boss data
        with open(f'plots_with_boss_{episode}_episodes/report.json', 'r') as f:
            with_boss_data = json.load(f)
        
        # Extract means and medians for each index
        for idx in indices:
            data['no_boss'][idx]['mean'].append(no_boss_data['stereotype_indices'][idx]['mean'])
            data['no_boss'][idx]['median'].append(no_boss_data['meta_statistics'][f'{idx}_stats']['median'])
            data['with_boss'][idx]['mean'].append(with_boss_data['stereotype_indices'][idx]['mean'])
            data['with_boss'][idx]['median'].append(with_boss_data['meta_statistics'][f'{idx}_stats']['median'])
    
    # Create subplot for each index
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Stereotype Indices Comparison: With Boss vs No Boss', fontsize=16)
    
    for idx, ax in zip(indices, axes.flat):
        ax.plot(episodes, data['with_boss'][idx]['mean'], 'b-', label='With Boss (Mean)', marker='o')
        ax.plot(episodes, data['with_boss'][idx]['median'], 'b--', label='With Boss (Median)', marker='s')
        ax.plot(episodes, data['no_boss'][idx]['mean'], 'r-', label='No Boss (Mean)', marker='o')
        ax.plot(episodes, data['no_boss'][idx]['median'], 'r--', label='No Boss (Median)', marker='s')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Index Value')
        ax.set_title(f'{idx.upper()} Comparison')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    fig.savefig('plots_all/boss_comparison.png')
def idx_meta_analysis_model():
    """Plot comparison of stereotype indices between different models"""
    # Group models by series
    model_groups = {
        'GPT': ['gpt-4o', 'gpt-4o-mini'],
        'Claude': [ 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest'],
        'Mistral': ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
        'Gemini': ['gemini-2.0-flash', 'gemini-1.5-flash']
    }
    
    # Flatten models list while maintaining group order
    models = []
    for group in model_groups.values():
        models.extend(group)
        
    indices = ['rsi', 'gbc', 'cai', 'sii']
    
    # Initialize data structures
    data = {model: {idx: {'mean': None, 'median': None} for idx in indices} for model in models}
    
    # Load data from reports
    for model in models:
        try:
            with open(f'plots_{model}/report.json', 'r') as f:
                model_data = json.load(f)
            
            # Extract means and medians for each index
            for idx in indices:
                data[model][idx]['mean'] = model_data['stereotype_indices'][idx]['mean']
                data[model][idx]['median'] = model_data['meta_statistics'][f'{idx}_stats']['median']
        except FileNotFoundError:
            print(f"Warning: No data found for {model}")
            continue
    
    # Create subplot for each index
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Stereotype Indices Comparison Across Models', fontsize=16)
    
    # Define color scheme for each model group
    group_colors = {
        'GPT': plt.cm.Reds(np.linspace(0.5, 0.8, len(model_groups['GPT']))),
        'Claude': plt.cm.Blues(np.linspace(0.5, 0.8, len(model_groups['Claude']))),
        'Mistral': plt.cm.Greens(np.linspace(0.5, 0.8, len(model_groups['Mistral']))),
        'Gemini': plt.cm.Purples(np.linspace(0.5, 0.8, len(model_groups['Gemini'])))
    }
    
    for idx, ax in zip(indices, axes.flat):
        x_pos = 0  # Starting position for bars
        xticks = []  # Store positions for x-axis labels
        xtick_labels = []  # Store labels for x-axis
        
        for group_name, group_models in model_groups.items():
            valid_models = [m for m in group_models if data[m][idx]['mean'] is not None]
            
            if not valid_models:
                continue
                
            means = [data[m][idx]['mean'] for m in valid_models]
            medians = [data[m][idx]['median'] for m in valid_models]
            
            width = 0.35
            x = np.arange(len(valid_models)) + x_pos
            
            # Plot bars for this group
            ax.bar(x - width/2, means, width, label=f'{group_name} Mean', 
                  color=group_colors[group_name])
            ax.bar(x + width/2, medians, width, label=f'{group_name} Median',
                  color=group_colors[group_name], alpha=0.7)
            
            # Store positions and labels
            xticks.extend(x)
            xtick_labels.extend(valid_models)
            
            # Update position for next group, adding a gap
            x_pos = x[-1] + 2
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Index Value')
        ax.set_title(f'{idx.upper()} Comparison')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        ax.grid(True, axis='y')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig('plots_all/model_comparison.png')

import os
def main(log_type, output_dir, s1_min, s1_max, s2_min, s2_max, model_name=None):
    import matplotlib
    matplotlib.use('Agg')
    
    meta = MetaAnalysis('../logs')
    # meta.load_experiments(pattern='*0s2*.csv')
    meta.load_experiments(s1_min, s1_max, s2_min, s2_max, model_name)
    meta.calculate_meta_statistics()
    os.makedirs(f'{output_dir}', exist_ok=True)
    meta.plot_meta_distributions().savefig(f'{output_dir}/meta_distributions.png')
    meta.plot_wc_space_clustering().savefig(f'{output_dir}/meta_wc_clustering.png')
    meta.plot_stereotype_indices_dist().savefig(f'{output_dir}/stereotype_indices_dist.png')
    meta.plot_indices_correlation().savefig(f'{output_dir}/indices_correlation.png')
    meta.plot_indices_boxplot().savefig(f'{output_dir}/indices_boxplot.png')
    meta.plot_indices_by_job_type().savefig(f'{output_dir}/indices_by_job_type.png')
    meta.plot_radar_chart().savefig(f'{output_dir}/radar_chart.png')
    
    # 生成报告
    report = meta.generate_report()
    print("\nMeta Analysis Report:")
    with open(f'{output_dir}/report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


MAX_EPISODES = 10000

if __name__ == "__main__":
    main('all', 'plots_all', 0, MAX_EPISODES, 0, MAX_EPISODES)
    main('no_boss', 'plots_no_boss', 0, MAX_EPISODES, 0, 0)
    main('with_boss', 'plots_with_boss', 0, MAX_EPISODES, 1, MAX_EPISODES)
    main('no_boss_10_episodes', 'plots_no_boss_10_episodes', 10, 10, 0, 0)
    main('no_boss_13_episodes', 'plots_no_boss_13_episodes', 13, 13, 0, 0)
    main('no_boss_16_episodes', 'plots_no_boss_16_episodes', 16, 16, 0, 0)
    main('no_boss_20_episodes', 'plots_no_boss_20_episodes', 20, 20, 0, 0)
    main('no_boss_25_episodes', 'plots_no_boss_25_episodes', 25, 25, 0, 0)
    main('with_boss_10_episodes', 'plots_with_boss_10_episodes', 6, 6, 4, 4)
    main('with_boss_13_episodes', 'plots_with_boss_13_episodes', 8, 8, 5, 5)
    main('with_boss_16_episodes', 'plots_with_boss_16_episodes', 10, 10, 6, 6)
    main('with_boss_20_episodes', 'plots_with_boss_20_episodes', 12, 12, 8, 8)
    main('with_boss_25_episodes', 'plots_with_boss_25_episodes', 15, 15, 10, 10)
    main('all_10_episodes', 'plots_all_10_episodes', 6, 10, 0, 4)
    main('all_13_episodes', 'plots_all_13_episodes', 8, 13, 0, 5)
    main('all_16_episodes', 'plots_all_16_episodes', 10, 16, 0, 6)
    main('all_20_episodes', 'plots_all_20_episodes', 15, 20, 0, 10)
    main('all_25_episodes', 'plots_all_25_episodes', 15, 25, 0, 10)

    main('gpt-4o', 'plots_gpt-4o', 0, MAX_EPISODES, 0, MAX_EPISODES, 'gpt-4o')
    main('gpt-4o-mini', 'plots_gpt-4o-mini', 0, MAX_EPISODES, 0, MAX_EPISODES, 'gpt-4o-mini')
    main('claude-3-5-haiku-latest', 'plots_claude-3-5-haiku-latest', 0, MAX_EPISODES, 0, MAX_EPISODES, 'claude-3-5-haiku-latest')
    main('claude-3-5-sonnet-latest', 'plots_claude-3-5-sonnet-latest', 0, MAX_EPISODES, 0, MAX_EPISODES, 'claude-3-5-sonnet-latest')
    main('mistral-large-latest', 'plots_mistral-large-latest', 0, MAX_EPISODES, 0, MAX_EPISODES, 'mistral-large-latest')
    main('mistral-medium-latest', 'plots_mistral-medium-latest', 0, MAX_EPISODES, 0, MAX_EPISODES, 'mistral-medium-latest')
    main('mistral-small-latest', 'plots_mistral-small-latest', 0, MAX_EPISODES, 0, MAX_EPISODES, 'mistral-small-latest')
    main('gemini-2.0-flash', 'plots_gemini-2.0-flash', 0, MAX_EPISODES, 0, MAX_EPISODES, 'gemini-2.0-flash')
    main('gemini-1.5-flash', 'plots_gemini-1.5-flash', 0, MAX_EPISODES, 0, MAX_EPISODES, 'gemini-1.5-flash')    
    # add boss comparison plot
    idx_meta_analysis_boss()
    
    # Add model comparison plot
    idx_meta_analysis_model()
