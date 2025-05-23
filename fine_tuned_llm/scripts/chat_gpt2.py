import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat_interface():
    # Load the fine-tuned model and tokenizer
    model_path = "./models/gpt2_finetuned"
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    print("\nChat with the fine-tuned GPT-2 model!")
    print("Type 'exit' to quit the chat.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        # Generate and print response
        try:
            response = generate_response(model, tokenizer, user_input)
            print("\nModel:", response)
        except Exception as e:
            print(f"\nError generating response: {str(e)}")

if __name__ == "__main__":
    chat_interface() 