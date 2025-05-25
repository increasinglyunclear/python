# Test script for philosophical analysis, taking output from VLM to fine-tuned LLM (see their respective repos)
# Kevin Walker May 2025

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Path to the VLM output
vlm_output_path = 'single_frame_analysis.txt'
# Path to the fine-tuned GPT-2 model
gpt2_model_dir = '/Users/kevin/Desktop/DP/philosophical_llm/models/gpt2_finetuned/checkpoint-90'
# Path to save the GPT-2 analysis
output_path = '/Users/kevin/Desktop/DP/philosophical_llm/vlm_gpt2_analysis.txt'

# 1. Read the VLM output
with open(vlm_output_path, 'r') as f:
    prompt = f.read()

# 2. Load the fine-tuned GPT-2 model and tokenizer
print('Loading fine-tuned GPT-2 model...')
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_dir)
model = GPT2LMHeadModel.from_pretrained(gpt2_model_dir)
model.eval()

# 3. Encode the prompt and generate a response
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Save the response
with open(output_path, 'w') as f:
    f.write(response)

print(f"GPT-2 analysis saved to {output_path}") 
