import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt






def visualize_times(times_df, model_name, audio_mode):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=times_df, x="Label", y="t_total", palette="Set2")
    plt.title("Boxplot of time taken for each screw type Model: " + model_name.upper() + (" (With Audio)" if audio_mode else " (No Audio)"))
    plt.xlabel("Screw Type")
    plt.ylabel("Total Time (s)")
    plt.tight_layout()
    

    return fig

def main():
    # Load the times data
    csv_path = Path(__file__).parent / "timeResults"

    for file in csv_path.glob('inference*.csv'):
        times_df = pd.read_csv(file)
        model_name = file.stem.split('_')[2]  # Extract model name from filename
        audio_mode = True if "med_audio" in file.stem else False  # Check if it's with audio

        fig = visualize_times(times_df, model_name, audio_mode)
        fig.savefig(Path(f"timeResults/inference_times_boxplot_{model_name}_{'med' if audio_mode else 'uden'}_audio_300_tests.png"))
    
    plt.show()

if __name__ == "__main__":
    main()

