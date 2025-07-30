
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_performance_chart(csv_filepath, output_image_path):
    """
    Loads evaluation data from a CSV and creates a grouped bar chart.

    Args:
        csv_filepath (str): The path to the master results CSV file.
        output_image_path (str): The path to save the generated chart image.
    """
    try:
        # Load the consolidated results from your CSV file
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        print("Please create the CSV file with your results first.")
        # As a fallback for demonstration, create a sample DataFrame
        data = {
            'Language Tier': ['High-Resource', 'High-Resource', 'High-Resource', 'High-Resource',
                              'Medium-Resource', 'Medium-Resource', 'Medium-Resource', 'Medium-Resource', 
                              'Low-Resource', 'Low-Resource', 'Low-Resource', 'Low-Resource'],
            'Model': ['gemma-3-1b-it', 'Mistral-7B-Instruct', 'Llama-3.1-8B-Instruct', 'Qwen3-8B',
                      'gemma-3-1b-it', 'Mistral-7B-Instruct', 'Llama-3.1-8B-Instruct', 'Qwen3-8B',
                      'gemma-3-1b-it', 'Mistral-7B-Instruct', 'Llama-3.1-8B-Instruct', 'Qwen3-8B'],
            'Accuracy': [0.404, 0.5712, 0.494, 0.4821,
                         0.355, 0.503, 0.467, 0.4482,
                         0.339, 0.418, 0.42, 0.390]
        }
        df = pd.DataFrame(data)

    # Set the style for the plot for a professional look
    sns.set_theme(style="whitegrid")

    # Create a figure and axes for the plot
    plt.figure(figsize=(12, 7))
    
    # The core plotting command 
    ax = sns.barplot(data=df, x='Language Tier', y='Accuracy', hue='Model', palette="viridis")

    # --- Customize the plot for clarity ---
    ax.set_title('Model Performance by Language Resource Tier', fontsize=16, weight='bold')
    ax.set_xlabel('Language Resource Tier', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    
    # Format the Y-axis to show percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Set the Y-axis limit to go up to 100%
    ax.set(ylim=(0, 1))

    # Add data labels on top of each bar for readability
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=10)

    # Improve the legend
    plt.legend(title='Language Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout 
    plt.tight_layout()

    plt.savefig(output_image_path)
    
    print(f"Chart successfully saved to {output_image_path}")


if __name__ == "__main__":
    results_csv = 'master_results.csv'
    chart_output = 'model_performance_chart.png'
    
    # Create the chart
    create_performance_chart(results_csv, chart_output)
