import os
import json
import torch
import argparse
from tqdm import tqdm

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_VICUNA = {
    "prompt_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction}\nInput:\n{input} ASSISTANT:"
    ),
    "prompt_no_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='vicuna', help='vicuna, wiz, alpaca')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    except:
        return 0, 0

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()
    
    except:
        return 0, 0



def main():

    args = parse_args()
    print(args)

    from transformers import LlamaTokenizer, LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
    sampled_data = sampled_data[exsisting_num:]


    if args.prompt == 'alpaca':
        prompt_no_input = PROMPT_DICT_ALPACA["prompt_no_input"]
        prompt_input = PROMPT_DICT_ALPACA["prompt_input"]
    elif args.prompt == 'wiz':
        prompt_no_input = PROMPT_DICT_WIZARDLM["prompt_no_input"]
        prompt_input = PROMPT_DICT_WIZARDLM["prompt_input"]
    elif args.prompt == 'vicuna':
        prompt_no_input = PROMPT_DICT_VICUNA["prompt_no_input"]
        prompt_input = PROMPT_DICT_VICUNA["prompt_input"]

    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['instruction']
        output_i = data_i['output']
        instruct_i_ori = instruct_i

        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        if input_i == '':
            temp_dict = {'instruction':instruct_i}
            promt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use

        else:
            temp_dict = {'instruction':instruct_i,'input':input_i}
            promt_to_use = prompt_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use

        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1] 


        instruct_i_reverse = 'Below is the response to an instruction, please guess the corresponding instruction for the given response.\n'+ output_i +'\nGenerate the instruction for the above response.'
        temp_dict_reverse = {'instruction':instruct_i_reverse}
        promt_to_use_reverse = prompt_no_input.format_map(temp_dict_reverse)
        whole_text_reverse = promt_to_use_reverse + instruct_i_ori + input_i
        instruct_i_reverse = promt_to_use_reverse

        instruct_i_reverse_input_ids = tokenizer.encode(instruct_i_reverse, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        instruct_i_reverse_len = instruct_i_reverse_input_ids.shape[1] 

        if output_i == '':
            temp_data_i = {}
        else:
            # Direct insruction and direct response
            ppl_ins_alone, loss_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, instruct_i_ori+input_i, args.max_length-instruct_i_reverse_len+1)
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, args.max_length-instruct_i_len+1)

            # Condition insruction and condition response
            ppl_ins_condition, loss_ins_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_reverse, instruct_i_ori, args.max_length)
            ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

            temp_data_i = {}
            temp_data_i['ppl'] = [ppl_ins_alone,ppl_out_alone,ppl_ins_condition,ppl_out_condition]
            temp_data_i['loss'] = [loss_ins_alone,loss_out_alone,loss_ins_condition,loss_out_condition]

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')

        pass

if __name__ == "__main__":
    main()