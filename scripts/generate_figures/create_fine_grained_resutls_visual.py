
# generate fine-grained plots for your final report and presentation.

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def parse_results(results_directory):
  
    all_results = []
    
    search_path = os.path.join(results_directory, "**", "*.json")
    
    print(f"Recursively searching for result files in: {results_directory}")
    
    #  find all matching files.
    for file_path in glob.glob(search_path, recursive=True):
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract Info from filename and JSON content
        if "hr-" in filename.lower() or "high_resource" in filename.lower():
            tier = "High-Resource"
        elif "mr-" in filename.lower() or "medium_resource" in filename.lower():
            tier = "Medium-Resource"
        elif "lr-" in filename.lower() or "low_resource" in filename.lower():
            tier = "Low-Resource"
        else:
            continue

        if "nouns" in filename.lower():
            pos = "Nouns"
        elif "verbs" in filename.lower():
            pos = "Verbs"
        elif "adjs" in filename.lower():
            pos = "Adjectives"
        else:
            continue

        task_name = list(data["results"].keys())[0]
        model_name = data["model_name"].split("/")[-1]
        accuracy = data["results"][task_name]["acc,none"]
        num_samples = data["n-samples"][task_name]["effective"]
        
        all_results.append({
            "Model": model_name,
            "Language Tier": tier,
            "Part of Speech": pos,
            "Accuracy": accuracy,
            "Number of Samples": num_samples
        })

    if not all_results:
        print("Warning: No result files were found or parsed. Please check your results directory and filenames.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_results)
    print("Successfully parsed the following data:")
    print(df)
    return df

def plot_fine_grained_performance(df, output_path):

    if df.empty:
        print("Cannot generate plot because no data was found.")
        return

    tier_order = ["High-Resource", "Medium-Resource", "Low-Resource"]
    df['Language Tier'] = pd.Categorical(df['Language Tier'], categories=tier_order, ordered=True)
    
    sns.set_theme(style="whitegrid", font_scale=1.1)

    g = sns.catplot(
        data=df,
        x='Language Tier',
        y='Accuracy',
        hue='Model',
        col='Part of Speech',
        kind='bar',
        palette='viridis',
        height=6,
        aspect=1.2
    )

    g.fig.suptitle('Fine-Grained Model Performance on MSI Benchmark', fontsize=20, weight='bold', y=1.03)
    g.set_axis_labels("Language Resource Tier", "Accuracy")
    g.set_titles("{col_name}", size=14)
    g.despine(left=True)
    
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set(ylim=(0, 1))
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1%}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    
    print(f"\nFine-grained performance chart saved to {output_path}")

if __name__ == "__main__":
    #  the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # parent directory (the project root)
    project_root = os.path.dirname(script_dir)
    
    # Build the full and absolute path to the results folder
    results_directory = os.path.join(project_root, 'results', 'Fine-grained - Nouns-Verbs-Adjs')
    
    results_df = parse_results(results_directory)

    if not results_df.empty:
        csv_output_path = os.path.join(project_root, "master_results_fine_grained.csv")
        results_df.to_csv(csv_output_path, index=False)
        print(f"\nMaster results table saved to {csv_output_path}")

        chart_output_path = os.path.join(project_root, "fine_grained_performance.png")
        plot_fine_grained_performance(results_df, chart_output_path)
