import os
import pandas as pd


PATH=r"C:\github\VT2\data\020320261"

def extract_csv_data(path):
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    data_frames = []
    
    for file in csv_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    return data_frames
def extract_json_data(path):
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    data_frames = []
    
    for file in json_files:
        file_path = os.path.join(path, file)
        df = pd.read_json(file_path)
        data_frames.append(df)
    
    return data_frames
if __name__ == "__main__":
    csv_data = extract_csv_data(PATH)
    json_data = extract_json_data(PATH)
    
    print("Extracted CSV Data:")
    for df in csv_data:
        print(df.head())
        print("\n")  # Add a newline for better separation between dataframes
        print([df.columns for df in csv_data])  # Print column names for each CSV dataframe
    
    print("\nExtracted JSON Data:")
    for df in json_data:
        print(df.head())
        print([df.columns for df in json_data])  # Print column names for each JSON dataframe