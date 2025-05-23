from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model configuration
    model_name = "meta-llama/Llama-2-7b-hf"  # We'll use 7B for M4 MacBook
    max_length = 2048
    batch_size = 4
    gradient_accumulation_steps = 4
    
    # Training configuration
    num_train_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.03
    
    # LoRA configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "v_proj"]
    
    # Output configuration
    output_dir = "../models/fine_tuned"
    logging_steps = 10
    save_steps = 100
    
    # Dataset configuration
    train_test_split = 0.9
    max_samples = None  # Set to None to use all samples
    
    # Generation configuration
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.2 