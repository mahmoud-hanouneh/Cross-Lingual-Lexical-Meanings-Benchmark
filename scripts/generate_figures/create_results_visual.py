import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_results(results_directory):

    # Parses all .json files in a directory to extract key evaluation results.
    
    # It infers the language tier from the filename 
  
    all_results = []
    
    print(f"Searching for result files in: {results_directory}")
    for filename in os.listdir(results_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(results_directory, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # ----- Extract Information -------
            # Infer tier from filename
            if filename.startswith("hr_"):
                tier = "High-Resource"
            elif filename.startswith("mr_"):
                tier = "Medium-Resource"
            elif filename.startswith("lr_"):
                tier = "Low-Resource"
            else:
                tier = "Unknown"

            # The task name can vary, so we get it from the results dictionary key
            task_name = list(data["results"].keys())[0]
            # model_name = data["config"]["model_name"].split("/")[-1] # Get a cleaner model name
            model_name = data["model_name"].split("/")[-1] 

            accuracy = data["results"][task_name]["acc,none"]
            num_samples = data["n-samples"][task_name]["effective"]
            
            all_results.append({
                "Model": model_name,
                "Language Tier": tier,
                "Accuracy": accuracy,
                "Number of Samples": num_samples
            })

    if not all_results:
        print("Warning: No result files were found or parsed. Please check your results directory and filenames.")
        return pd.DataFrame()
        
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_results)
    print("Successfully parsed the following data:")
    print(df)
    return df

def plot_performance_by_tier(df, output_path):
    """
    Creates a grouped bar chart comparing model performance across language tiers.

    """
    if df.empty:
        print("Cannot generate 'performance_by_tier' plot because no data was found.")
        return

    # Order the tiers for the plot
    tier_order = ["High-Resource", "Medium-Resource", "Low-Resource"]
    df['Language Tier'] = pd.Categorical(df['Language Tier'], categories=tier_order, ordered=True)
    df = df.sort_values('Language Tier')

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x='Language Tier', y='Accuracy', hue='Model', palette='viridis')

    ax.set_title('Model Performance on MSI Benchmark by Language Tier', fontsize=16, weight='bold')
    ax.set_xlabel('Language Resource Tier', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.set(ylim=(0, 1))

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)

    plt.legend(title='Language Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nGrouped performance chart saved to {output_path}")

# Dataset Size chart 
def plot_dataset_size(df, output_path):
   
    if df.empty:
        print("Cannot generate 'dataset_size' plot because no data was found.")
        return
        
    # Get the number of samples per tier (drop duplicates to count each tier once)
    size_df = df[['Language Tier', 'Number of Samples']].drop_duplicates().sort_values('Language Tier')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=size_df, x='Language Tier', y='Number of Samples', palette='plasma')

    ax.set_title('Number of Generated Benchmark Examples per Tier', fontsize=16, weight='bold')
    ax.set_xlabel('Language Resource Tier', fontsize=12)
    ax.set_ylabel('Number of Questions (Prompts)', fontsize=12)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Dataset size chart saved to {output_path}")

# --- Main execution ---
if __name__ == "__main__":
    # Define the directory where your JSON result files are located
    RESULTS_DIR = '././results/Final Results' 
    
    # Consolidate all results into a single DataFrame
    results_df = parse_results(RESULTS_DIR)

    if not results_df.empty:
        # Save the consolidated data to a master CSV file for easy access
        results_df.to_csv("master_results.csv", index=False)
        print(f"\nMaster results table saved to master_results.csv")

        # Generate the plots
        plot_performance_by_tier(results_df, "performance_by_tier.png")
        plot_dataset_size(results_df, "dataset_size.png")
