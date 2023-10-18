import json
import openai
import string
import argparse
import re

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

def gen_prompt_no_input(ins, outp):

    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of a given instruction. \n" + \
                "1. Why this instruction is not good? First analyse the instruction based on Complexity of the Topic, Level of Detail Required, Knowledge Required, Ambiguity of the Instruction and Logical Reasoning or Problem-Solving Involved. \n" + \
                "Then analyse why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "Finally analyse why this bad instruction lead to a bad answer. " +\
                "2. Based on the reason you provided, generate a new and complete instruction which is complex and difficult to answer directly. " + \
                "Make sure the new instruction is relevent but independent to the original instruction, which can be answered without knowing the original instruction, put the new instruction in the format of [New Instruction] your instruction [End]" +\
                "3. Answer the newly generated instruction as detailed as possible, in the format of [New Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt


def extract_ins(text,no_input=True):
    if '[New Instruction]' in text:
        pattern = r'(\[New Instruction\])(.*?)(\[End\]|\[New Answer\]|New Answer:)'
    else:
        pattern = r'(New Instruction:)(.*?)(\[End\]|\[New Answer\]|New Answer:)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_ins = ''
    else:
        seg_ins = segments[0][1].strip()
    if seg_ins.endswith("\n\n3."):
        seg_ins = seg_ins[:-4]
    return seg_ins

def extract_oup(text,no_input=True):
    if '[New Answer]' in text:
        pattern = r'(\[New Answer\])(.*?)(\[End\]|$)'
    else:
        pattern = r'(New Answer:)(.*?)(\[End\]|$)'
        # pattern = r'(\[New Answer\]|New Answer:)(.*?)(\[End\]|$)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_oup = ''
    else:
        seg_oup = segments[0][1].strip()
    return seg_oup

def extract_segments_no_input(text):
    if text == '':
        return []
    seg_ins = extract_ins(text,no_input=True)
    seg_oup = extract_oup(text,no_input=True)
    return [seg_ins,seg_oup]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default='')
    parser.add_argument("--ori_data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--save_intermediate_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="maximum number of tokens produced in the output",
    )

    args = parser.parse_args()
    openai.api_key = args.api_key
    model_engine = args.api_model

    raw_json_data_path = args.raw_data_path
    with open(raw_json_data_path,'r') as f:
        raw_data = json.load(f)

    ori_json_data_path = args.ori_data_path
    with open(ori_json_data_path,'r') as f:
        ori_data = json.load(f)

    new_data = []
    retry_num = 0
    for i, raw_data_i in enumerate(raw_data):

        if (i+1) % 1000 == 0:
            print(i+1,'/',len(raw_data))

        ori_data_i = ori_data[i]
        instruct_i = ori_data_i['instruction'].strip()
        output_i = ori_data_i['output'].strip()
        if 'input' in ori_data_i.keys():
            input_i = ori_data_i['input'].strip()
        else:
            ori_data_i['input'] = ''
            input_i = ''
            
        retry_flag = False
        seg_list = extract_segments_no_input(raw_data_i)
        if len(seg_list) != 2:
            retry_flag = True
        else:
            if seg_list[0] == '' and seg_list[1] == '':
                retry_flag = True
            if (seg_list[0] == '') or ('your instruction' in seg_list[0]):
                seg_list[0] = instruct_i
            if ('N/A' in seg_list[1]) or (seg_list[1]=='') or ('your answer' in seg_list[1]):
                seg_list[1] = output_i

        if retry_flag:
            retry_num += 1

            sys_prompt, prompt = gen_prompt_no_input(instruct_i, output_i)
            
            token_limit = min(args.max_tokens,4050-len(gpt_encoder.encode(prompt)))
            response = ''
            try:
                message =[
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                ]
                completion = openai.ChatCompletion.create(
                            model=model_engine,
                            messages=message,
                            temperature=0.0,
                            max_tokens=token_limit,
                            top_p=1.0,
                )
                response = completion.choices[0].message.content
            except:
                response = ''

            seg_list = extract_segments_no_input(response)
            # seg_list = [x for x in seg_list if x != '']


        temp_data = {}
        temp_data['instruction'] = ori_data_i['instruction']
        temp_data['output'] = ori_data_i['output']
        temp_data['input'] = ori_data_i['input']

        if len(seg_list) != 2:
            temp_data['new_instruct'] = ori_data_i['instruction']
            temp_data['new_answer'] = ori_data_i['output']
        else:
            if (seg_list[0] == '') or ('your instruction' in seg_list[0]):
                temp_data['new_instruct'] = ori_data_i['instruction']
            else:
                temp_data['new_instruct'] = seg_list[0]

            if ('N/A' in seg_list[1]) or (seg_list[1]=='') or ('your answer' in seg_list[1]):
                temp_data['new_answer'] = ori_data_i['output']
            else:
                temp_data['new_answer'] = seg_list[1]

        temp_data['new_input'] = ''
        new_data.append(temp_data)

        pass
    print('retry_num',retry_num)
    if args.save_intermediate_path != '':
        with open(args.save_intermediate_path,'w') as f:
            json.dump(new_data,f,indent=4)
    
    final_new_data = []
    none_count = 0
    for i, data_i in enumerate(new_data):
        temp_data = {}

        if (data_i['new_instruct'] == '') and (data_i['new_answer'] == ''):
            none_count += 1
            temp_data['instruction'] = data_i['instruction']
            temp_data['output'] = data_i['output']
            temp_data['input'] = data_i['input']
        else:
            temp_data['instruction'] = data_i['new_instruct']
            temp_data['output'] = data_i['new_answer']
            temp_data['input'] = data_i['new_input'] 

        final_new_data.append(temp_data)

    print('none_num',none_count)
    print('Len New Data', len(final_new_data))
    with open(args.save_path,'w') as f:
        json.dump(final_new_data,f,indent=4)