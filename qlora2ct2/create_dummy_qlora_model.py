import torch
from peft import PeftModel,  LoraConfig, get_peft_model
from transformers import  BitsAndBytesConfig,  AutoTokenizer, AutoModelForSeq2SeqLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", quantization_config=bnb_config, device_map="auto", use_cache=False, cache_dir = "./cache")

model.gradient_checkpointing_enable()
# model = prepare_model_for_4bit_training(model)

config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules=["q", "v"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="SEQ_2_SEQ_LM"
)
print(config)

model = get_peft_model(model, config)

model.save_pretrained("./adapter")