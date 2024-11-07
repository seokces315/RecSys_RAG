from transformers import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import torch


# Function to load base model with given model_id
def load_base_model(model):
  
    # 4-bit quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    #base_model.config.use_cache = False
    
    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    return base_model, tokenizer
