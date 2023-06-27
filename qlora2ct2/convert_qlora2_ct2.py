import torch
from peft import PeftModel, PeftConfig
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import argparse
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def convert_qlora2ct2(config_path = "convert_config_with_quantization.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    adapter_path = config["adapter_path"]
    full_model_path = config["full_model_path"]
    offload_path = config["offload_path"]
    ct2_path = config["ct2_path"]
    quantization = config["quantization"]

    peft_model_id = adapter_path
    peftconfig = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(peftconfig.base_model_name_or_path, 
                                                device_map = "auto", 
                                                offload_folder  = offload_path
                                                )

    tokenizer = AutoTokenizer.from_pretrained(peftconfig.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id)

    print("Peft model loaded")

    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(full_model_path) 
    tokenizer.save_pretrained(full_model_path)

    if quantization == False:
        os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --force")
    else:
        os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --quantization {quantization} --force")
    print("Convert successfully")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Convert config')
    parser.add_argument('convert_config_path', 
                    default="convert_config_with_quantization.json",
                    )
    args = parser.parse_args()
    # print(args.convert_config_path)
    convert_qlora2ct2(args.convert_config_path)