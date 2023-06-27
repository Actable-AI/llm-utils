This is the trick for faster inference of qlora model using the Ctranslate2 library instead of normal inference.

The method basically just merges the adapter of qlora model with the full model, saves it, and then converts it to ct2.

If ct2 supports your base model, then this method works just fine.

Steps to convert qlora model to the Ctranslate2 model:
1. Load your base model (you can know the base model name by checking in the adapter_config.json file) using HuggingFace
2. Get the peft model from your base model and the adapter model path
3. Merge and unload the peft model to get the full model
4. Save the full model and the tokenizer in the full model path
5. After you have the full model path, convert to Ctranslate2 format using the script in the convert.sh file

Note: there are some tricks
- You can load your base model using torch_dtype = torch.bfloat16 to decrease storage two times
- When converting, you can use quantization to further decrease storage (for example, using int8_float16 quantization decreases storage 4 times (48GB-->12GB) when converting mt0 xxl model) --> so you can both use torch_dtype = torch.bfloat16 and int8_float16 quantization
- When doing inference with ct2 model, you should use the compute_type same as the type of quantization when you convert, or just use int8 compute_type

PS: If you find this useful, please leave a star :))
