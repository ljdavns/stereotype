import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import networkx as nx
from sklearn.metrics import cohen_kappa_score
import ast
import os
from adjustText import adjust_text
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from tools.jobs_less import jobs

def load_and_process_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Get rating data from the last episode
    last_episode = df['episode'].max()
    ratings_df = df[
        (df['episode'] == last_episode) & 
        (df['action'] == 'discuss_parsed')
    ].copy()
    
    # Parse JSON data in the message field
    ratings_df['parsed_message'] = ratings_df['message'].apply(
        lambda x: ast.literal_eval(x)
    )
    
    return ratings_df

def create_person_job_matrix(ratings_df):
    # Create person-job matrix
    people = [f'person_{i}' for i in range(5)]
    job_names = [job['name'].lower() for job in jobs]
    
    matrix = pd.DataFrame(0, index=people, columns=job_names)
    
    for _, row in ratings_df.iterrows():
        person_view = row['parsed_message']['person_view']
        for person, jobs_list in person_view.items():
            for job in jobs_list:
                matrix.loc[person, job.lower()] += 1
    
    return matrix

def calculate_warmth_competence(matrix):
    # Calculate warmth and competence scores for each person
    people = matrix.index
    warmth_scores = pd.Series(0.0, index=people)
    competence_scores = pd.Series(0.0, index=people)
    
    job_info = {job['name'].lower(): job for job in jobs}
    
    for person in people:
        total_mentions = matrix.loc[person].sum()
        if total_mentions > 0:
            for job_name in matrix.columns:
                weight = matrix.loc[person, job_name] / total_mentions
                warmth_scores[person] += weight * job_info[job_name]['warmth(1-10)']
                competence_scores[person] += weight * job_info[job_name]['competence(1-10)']
    
    return warmth_scores, competence_scores

def plot_warmth_competence(warmth_scores, competence_scores, clusters=None):
    plt.figure(figsize=(10, 8))
    
    # If clusters are provided, use cluster coloring; otherwise use regular scatter plot
    if clusters is not None:
        scatter = plt.scatter(competence_scores, warmth_scores, 
                            c=clusters, cmap='viridis', s=100)
        plt.colorbar(scatter)
    else:
        plt.scatter(competence_scores, warmth_scores, s=100)
    
    # Add labels and avoid overlapping
    texts = []
    for i, txt in enumerate(warmth_scores.index):
        texts.append(plt.text(competence_scores[i], warmth_scores[i], txt))
    
    # Use adjust_text to avoid label overlapping
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
               expand_points=(2, 2),
               force_points=(1, 1),
               force_text=(0.5, 0.5),
               expand_text=(1.5, 1.5),
               text_from_points=(0.5, 0.5),
               only_move={'points':'xy',
                         'text':'xy'})
    
    plt.xlabel('Competence Score')
    plt.ylabel('Warmth Score')
    plt.title('Warmth-Competence Analysis' + (' with Clusters' if clusters is not None else ''))
    
    # Add quadrant dividing lines
    plt.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_heatmap(matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Person-Job Association Heatmap')
    return plt.gcf()

def create_network_graph(matrix):
    # Create bipartite graph
    G = nx.Graph()
    
    # Add nodes
    people = list(matrix.index)
    jobs = list(matrix.columns)
    
    G.add_nodes_from(people, bipartite=0)
    G.add_nodes_from(jobs, bipartite=1)
    
    # Add edges
    for person in people:
        for job in jobs:
            weight = matrix.loc[person, job]
            if weight > 0:
                G.add_edge(person, job, weight=weight)
    
    # Draw network graph
    plt.figure(figsize=(12, 8))
    
    # Set layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=people, 
                          node_color='lightblue',
                          node_size=500)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=jobs, 
                          node_color='lightgreen',
                          node_size=500)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, 
                          width=[w/2 for w in weights],
                          alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Person-Job Network Analysis')
    plt.axis('off')
    return plt.gcf()

def calculate_statistics(matrix):
    # Calculate basic statistical indicators
    stats_dict = {
        'total_associations': matrix.sum().sum(),
        'mean_associations_per_person': matrix.sum(axis=1).mean(),
        'mean_associations_per_job': matrix.sum(axis=0).mean(),
        'std_associations_per_person': matrix.sum(axis=1).std(),
        'std_associations_per_job': matrix.sum(axis=0).std()
    }
    
    # Calculate most common jobs for each person
    most_common_jobs = matrix.idxmax(axis=1)
    stats_dict['most_common_jobs'] = most_common_jobs.to_dict()
    
    # Calculate most common people for each job
    most_common_people = matrix.idxmax(axis=0)
    stats_dict['most_common_people'] = most_common_people.to_dict()
    
    return stats_dict

def calculate_advanced_statistics(matrix, warmth_scores, competence_scores):
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # Basic statistical indicators
    stats_dict = {
        'total_associations': matrix.sum().sum(),
        'mean_associations_per_person': matrix.sum(axis=1).mean(),
        'mean_associations_per_job': matrix.sum(axis=0).mean(),
        'std_associations_per_person': matrix.sum(axis=1).std(),
        'std_associations_per_job': matrix.sum(axis=0).std()
    }
    
    # 1. Similarity Analysis
    # Calculate similarity matrix between people (using cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(matrix)
    stats_dict['mean_similarity'] = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    stats_dict['max_similarity'] = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    
    # 2. Variance Analysis
    # Test differences in job scores
    job_scores = matrix.values
    f_stat, p_value = stats.f_oneway(*[job_scores[:, i] for i in range(job_scores.shape[1])])
    stats_dict['job_anova_f'] = f_stat
    stats_dict['job_anova_p'] = p_value
    
    # 3. Correlation Analysis
    # Calculate correlation coefficient between warmth and competence (halo effect)
    correlation, p_value = stats.pearsonr(warmth_scores, competence_scores)
    stats_dict['warmth_competence_correlation'] = correlation
    stats_dict['warmth_competence_correlation_p'] = p_value
    
    # 4. Cluster Analysis
    # Use KMeans for clustering and calculate silhouette score
    X = np.column_stack((warmth_scores, competence_scores))
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    stats_dict['cluster_silhouette_score'] = silhouette_avg
    
    # 5. Consistency Analysis
    # Calculate coefficient of variation (CV) for each job rating
    cv_scores = matrix.std() / matrix.mean()
    stats_dict['job_cv'] = cv_scores.to_dict()
    
    # 6. Preference Strength Analysis
    # Calculate the difference between highest and second-highest scores for each person
    preference_strength = []
    for _, row in matrix.iterrows():
        sorted_scores = np.sort(row)[::-1]
        if len(sorted_scores) >= 2:
            preference_strength.append(sorted_scores[0] - sorted_scores[1])
    stats_dict['mean_preference_strength'] = np.mean(preference_strength)
    
    # 7. Information Entropy Analysis
    # Calculate information entropy of job distribution (lower values indicate stronger stereotypes)
    def entropy(probs):
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    job_probs = matrix / matrix.sum(axis=1).values.reshape(-1, 1)
    entropies = job_probs.apply(entropy, axis=1)
    stats_dict['mean_entropy'] = entropies.mean()
    
    return stats_dict

def plot_similarity_heatmap(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                cmap='YlOrRd', 
                xticklabels=similarity_matrix.columns,
                yticklabels=similarity_matrix.index)
    plt.title('Person-Job Similarity Heatmap')
    return plt.gcf()

def main():
    # Set the display backend to avoid WSL hanging
    import matplotlib
    matplotlib.use('Agg')
    
    # Create plots directory (if it doesn't exist)
    os.makedirs('./plots', exist_ok=True)
    
    # Read data
    ratings_df = load_and_process_data('../logs/stereotype_multiagent_gpt-4o_5ppl_10s1_6s2_20250213_195846.csv')
    
    # Create person-job matrix
    matrix = create_person_job_matrix(ratings_df)
    
    # Calculate warmth and competence scores
    warmth_scores, competence_scores = calculate_warmth_competence(matrix)
    
    # Calculate basic statistical indicators
    basic_stats = calculate_statistics(matrix)
    print("\nBasic Statistical Analysis:")
    for key, value in basic_stats.items():
        print(f"{key}: {value}")
    
    # Calculate advanced statistical indicators
    advanced_stats = calculate_advanced_statistics(matrix, warmth_scores, competence_scores)
    print("\nAdvanced Statistical Analysis:")
    for key, value in advanced_stats.items():
        print(f"{key}: {value}")
    
    # Cluster analysis
    from sklearn.cluster import KMeans
    X = np.column_stack((warmth_scores, competence_scores))
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Generate warmth-competence plot with clusters
    wc_plot = plot_warmth_competence(warmth_scores, competence_scores, clusters)
    wc_plot.savefig('./plots/warmth_competence_plot.png', bbox_inches='tight')
    
    heatmap = plot_heatmap(matrix)
    heatmap.savefig('./plots/person_job_heatmap.png', bbox_inches='tight')
    
    # Calculate similarity matrix (Person-Job similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    # Normalize processing to avoid large numbers
    normalized_matrix = matrix / matrix.sum(axis=1).values.reshape(-1, 1)
    similarity_matrix = pd.DataFrame(
        normalized_matrix,  # Directly use normalized data
        index=matrix.index,     # Person labels
        columns=matrix.columns  # Job labels
    )
    
    # Generate similarity heatmap
    similarity_heatmap = plot_similarity_heatmap(similarity_matrix)
    similarity_heatmap.savefig('./plots/similarity_heatmap.png', bbox_inches='tight')

    # Print results
    print("\nWarmth Scores:")
    print(warmth_scores)
    print("\nCompetence Scores:")
    print(competence_scores)

if __name__ == "__main__":
    main()