import os
import pandas as pd

def create_dataset_csv(base_dir='img', styles=['dark', 'neon', 'pagan', 'tech']):
    # Initialize an empty list to hold our data entries
    data = []

    # Iterate through each style directory
    for style in styles:
        # Construct the full path to the style directory
        style_dir = os.path.join(base_dir, style)
        
        # Check if the directory exists to avoid errors
        if not os.path.exists(style_dir):
            print(f"Warning: Directory {style_dir} does not exist.")
            continue
        
        # Loop through all files in the style directory
        for filename in os.listdir(style_dir):
            # Check if the file is an image by checking the extension
            if filename.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                # Construct the full file path
                file_path = os.path.join(style_dir, filename)
                # Add the file path and its corresponding style to our data list
                data.append({'file_path': file_path, 'style': style})

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file without the index column
    df.to_csv('dataset.csv', index=False)
    print(f"Created dataset.csv with {len(df)} entries.")

if __name__ == "__main__":
    # Call the function to create the dataset when the script is run
    create_dataset_csv()