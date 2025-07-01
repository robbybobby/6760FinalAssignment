import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
from datasets import load_dataset

################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


""" def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer """

def load_model(model_name_or_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer

def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))        
        else:
            assert os.getenv("HF_TOKEN"), "HF_TOKEN is not set"
            dataset = load_dataset(f"cais/wmdp-{name}", split="train", token=os.getenv("HF_TOKEN"))
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

"""def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []

        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))

        elif name == "retain_qa":
            dataset = load_dataset("muse-bench/MUSE-News", 'knowmem', split="retain_qa")
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        
        elif name == "retain_qa_icl":
            dataset = load_dataset("muse-bench/MUSE-News", 'knowmem', split="retain_qa_icl")
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        
        elif name == "forget_qa":
            dataset = load_dataset("muse-bench/MUSE-News", 'knowmem', split="forget_qa")
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))

        elif name  == "forget_qa_icl":
            dataset = load_dataset("muse-bench/MUSE-News", 'knowmem', split="forget_qa_icl")
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        
        else:
            raise ValueError(f"Dataset name '{name}' not recognized")

        # Batch the data
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )"""