from peft import LoraConfig
from transformers import TrainingArguments

# Method to get parameter-efficient fine-tuning params
def get_peft_config(r, lora_alpha, lora_dropout):
  
    # Define lora config
    peft_config = LoraConfig(
        task_type="CASUAL_LM",
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none"
    )
    
    return peft_config
  
  
# Method to get training arguments for SFTTrainer
def get_training_args(per_device_train_batch_size, learning_rate, weight_decay,
                      max_grad_norm, num_train_epochs, lr_scheduler_type, warmup_ratio):
  
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        logging_dir="./logs",
        logging_steps=100,
        label_names=["target"],
        optim="paged_adamw_32bit",
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        report_to="tensorboard"
    )
    
    return training_args
  