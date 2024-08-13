from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from dataset import load_dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def main():
    # Define model repository
    model_repo = "EleutherAI/polyglot-ko-5.8b"

    # Set up 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=False)
    
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(model_repo, quantization_config=bnb_config)

    # Prepare model for LoRA with 4-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrix
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        task_type="CAUSAL_LM"  # Type of task
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # Load the dataset
    train_dataset = load_dataset(tokenizer_name=model_repo, file_path='/data/hyeokseung1208/cchat/data/consult_highschool.txt')

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We are doing causal language modeling, not masked language modeling
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="/data/hyeokseung1208/cchat/weights",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust as needed
        learning_rate=2e-4,
        per_device_train_batch_size=2,  # Adjust as needed
        prediction_loss_only=True,
        logging_dir='/data/hyeokseung1208/cchat/logs',
        fp16=True,  # Enable mixed precision training
        logging_steps=10,
        gradient_accumulation_steps=16,  # Helps with reducing memory usage
    )


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model("/data/hyeokseung1208/cchat/outputs")
    tokenizer.save_pretrained("/data/hyeokseung1208/cchat/outputs")

if __name__ == "__main__":
    main()