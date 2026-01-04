# import json
# import yaml

# with open('sphere_database-11.json', 'r') as f:
#     data = json.load(f)

# with open('sphere_database-11.yaml', 'w') as f:
#     yaml.dump(data, f, sort_keys=False)


import json
import yaml

# Load source data
with open('sphere_database-37.json', 'r') as f:
    data = json.load(f)

processed_data = {}

for mesh_key, nested_content in data.items():
    if "8" in nested_content:
        # Identify all sub-keys under "8"
        sub_indices = nested_content["8"].keys()
        
        # Determine the highest numerical key
        highest_key = str(max(int(k) for k in sub_indices))
        
        # Extract data associated with the highest key
        target_payload = nested_content["8"][highest_key]
        
        # Reconstruct structure: assign highest key's data to key "0"
        processed_data[mesh_key] = {
            "8": {
                "0": target_payload
            }
        }

# Export to YAML
with open('sphere_database-37.yaml', 'w') as f:
    yaml.dump(processed_data, f, sort_keys=False)