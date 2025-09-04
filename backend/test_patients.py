import os
import json

# Try different paths
paths = [
    'data/patients',
    'backend/data/patients',
    './data/patients',
    '../data/patients',
    './backend/data/patients',
    '../backend/data/patients'
]

for path in paths:
    print(f"Trying path: {path}")
    try:
        files = os.listdir(path)
        print(f"  Success! Found {len(files)} files")
        if files:
            print(f"  First file: {files[0]}")
            try:
                with open(f"{path}/{files[0]}", 'r') as f:
                    data = json.load(f)
                    print(f"  Successfully loaded file: {files[0]}")
                    print(f"  Keys: {list(data.keys())}")
            except Exception as e:
                print(f"  Error loading file: {str(e)}")
    except Exception as e:
        print(f"  Error: {str(e)}")
