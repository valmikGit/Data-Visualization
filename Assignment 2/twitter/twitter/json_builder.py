from pathlib import Path
import json

current_folder = Path(__file__).parent  # Gets the folder of the current script
file_path = current_folder / '81938351.feat'  # Construct the full path

# Check if the file exists before accessing it
if file_path.exists():
    print(f"The file exists at: {file_path}")
else:
    print(f"File not found: {file_path}")

# List only regular files, exclude symbolic links and hidden files
file_names = [
    file.name for file in current_folder.iterdir() 
    if file.is_file() and not file.name.startswith('.') and file.exists()
]

file_number_names = []

for filename in file_names:
    file_number_names.append(filename.strip().split('.')[0])

# Define a function to read each type of file and store in a dictionary
def parse_files(file_id) -> bool:
    data_dict = {}
    
    # Construct the file paths
    feat_file_path = current_folder / f"{file_id}.feat"
    featnames_file_path = current_folder / f"{file_id}.featnames"
    circles_file_path = current_folder / f"{file_id}.circles"
    edges_file_path = current_folder / f"{file_id}.edges"
    egofeat_file_path = current_folder / f"{file_id}.egofeat"

    try:
        # Example: Parsing .feat file
        with open(feat_file_path, "r") as feat_file:
            data_dict['features'] = [list(map(float, line.strip().split())) for line in feat_file]

        # Example: Parsing .featnames file
        with open(featnames_file_path, "r") as featnames_file:
            data_dict['feature_names'] = [line.strip() for line in featnames_file]

        # Example: Parsing .circles file
        with open(circles_file_path, "r") as featnames_file:
            data_dict['feature_names'] = [line.strip() for line in featnames_file]

        # Example: Parsing .edges file
        with open(edges_file_path, "r") as featnames_file:
            data_dict['feature_names'] = [line.strip() for line in featnames_file]

        # Example: Parsing .egofeat file
        with open(egofeat_file_path, "r") as featnames_file:
            data_dict['feature_names'] = [line.strip() for line in featnames_file]

        # Save parsed data to JSON
        with open(current_folder / f"{file_id}_data.json", "w") as json_file:
            json.dump(data_dict, json_file)
            return True
    except Exception as e:
        print(f"Error processing {file_id}: {e}")  # Print the error for debugging
        return False

# Example usage
file_ids = ['224581852', '19725644', '22252971']  # Example file IDs
for file_id in file_ids:
    parse_files(file_id)

for file_id in list(set(file_number_names)):
    status = parse_files(file_id)
    if status == False:
        continue

# Save the list of file names as unique file IDs
unique_file_ids = list(set(file_number_names))
unjsoned_files = []

for unique_file_name in unique_file_ids:
    if "_data" in unique_file_name:
        print("The string contains '_data'.")
    else:
        parse_files(unique_file_name)
        print("The string does not contain '_data'.")

# Write the list to a JSON file
with open(current_folder / 'file_ids.json', 'w') as f:
    json.dump(unique_file_ids, f)