import json
import os

def extract_narratives(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    narratives = []
    for frame in data['frame_analysis']:
        narratives.append(frame['narrative'])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(narratives))
    
    print(f"Narratives extracted and saved to {output_file}")

if __name__ == "__main__":
    json_file = "analysis_results/video_analysis_20250524_113801.json"
    output_file = "narratives.txt"
    extract_narratives(json_file, output_file) 