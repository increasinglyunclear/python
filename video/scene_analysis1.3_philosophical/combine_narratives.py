import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def combine_narratives(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    narratives = [frame['narrative'] for frame in data['frame_analysis']]
    combined_narrative = " ".join(narratives)
    
    # Generate a summary narrative using GPT-2
    prompt = f"Based on the following narratives from a video analysis, create a single coherent philosophical narrative:\n\n{combined_narrative}\n\nSummary Narrative:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=inputs.shape[1]+150, num_return_sequences=1, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the summary narrative (after 'Summary Narrative:')
    if 'Summary Narrative:' in summary:
        summary = summary.split('Summary Narrative:')[-1].strip()
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Combined narrative saved to {output_file}")

if __name__ == "__main__":
    json_file = "analysis_results/video_analysis_20250524_114629.json"
    output_file = "combined_narrative.txt"
    combine_narratives(json_file, output_file) 