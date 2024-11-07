import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data import join_dataset, generate_prompts, load_dataset
from model.llm import load_base_model
from model.train import get_peft_config, get_training_args
from utils import set_seed

from trl import SFTTrainer

import torch
from transformers import pipeline

HF_TOKEN = "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"

def main():

    # 1. Set seed for reproducibility
    set_seed(42)

    # 2. Load dataset
    category = "Toys_and_Games"
    test_size = 1 / 10.0
    rag_size = 8 / 9.0
    task_id_list = [
        "1-1",
        # "1-2",
        # "1-3",
        # "1-4",
        # "1-5",
        # "1-6",
    ]
    
    TG_dataset = join_dataset(category)
    PP_dict = generate_prompts(TG_dataset, test_size, rag_size, task_id_list)
    base_dataset = load_dataset(PP_dict)
    
    # 3. Load model
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    base_model, tokenizer = load_base_model(model)

    # 4. Get parameter-efficient fine-tuning params
    r = 8
    lora_alpha = 8
    lora_dropout = 0.0
    
    peft_config = get_peft_config(r, lora_alpha, lora_dropout)
    
    # 5. Get training arguments for SFT train
    per_device_train_batch_size = 8
    learning_rate = 5e-05
    weight_decay = 0.0
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    lr_scheduler_type = "linear"
    warmup_ratio = 0.0
    
    training_args = get_training_args(per_device_train_batch_size, learning_rate, weight_decay,
                                      max_grad_norm, num_train_epochs, lr_scheduler_type, warmup_ratio)
    
    # 6. Perform SFT training
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=base_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )
    
    trainer.train()
    
    print("Success!")
    print()

    # model_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     token=HF_TOKEN,
    # )
    
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a chatbot designed to perform rating predictions within a range of 1 to 5!",
    #     },
    #     {"role": "user", "content": base_dataset["text"][0]},
    # ]
    
    # outputs = model_pipeline(
    #     messages,
    #     max_new_tokens=256,
    # )

    # print()
    # print("[ Answer ]")
    # print(outputs[0]["generated_text"][-1]["content"])
    # print()

    return


if __name__ == "__main__":
    main()
