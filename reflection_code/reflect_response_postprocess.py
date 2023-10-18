import json
import openai
import string
import argparse
import re

def gen_prompt_no_input(ins, outp):

    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of the answer to a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction. \n" + \
                "1. Why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n" 
    prompt = prompt_template.format(
        ins=ins, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt

def gen_prompt_input(ins, inp, outp):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer to a given instruction and its input."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Input]\n{inp}\n\n[The End of Input]\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction and corresponding input. \n" + \
                "1. Why this answer is not good for the given instruction and corresponding input? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n" 
    prompt = prompt_template.format(
        ins=ins, inp=inp, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt


def extract_segments(text):
    if text.count('[Better Answer]') >= 2:
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|\[Better Answer\]|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    else:
        # pattern = r'\[(Better Answer)\](.*?)\[End\]'
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|End|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    return [segment[1].strip() for segment in segments]


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

    with open(args.raw_data_path,'r') as f:
        raw_data = json.load(f)

    with open(args.ori_data_path,'r') as f:
        ori_data = json.load(f)

    new_data = []
    for i, raw_data_i in enumerate(raw_data):
        if (i+1) % 1000 == 0:
            print(i+1,'/',len(raw_data))
        seg_list = extract_segments(raw_data_i)

        ori_data_i = ori_data[i]
        instruct_i = ori_data_i['instruction'].strip()
        output_i = ori_data_i['output'].strip()
        if 'input' in ori_data_i.keys():
            input_i = ori_data_i['input'].strip()
        else:
            input_i = ''

        if len(seg_list) != 1:

            if input_i == '':
                sys_prompt, prompt = gen_prompt_no_input(instruct_i, output_i)
            else:
                sys_prompt, prompt = gen_prompt_input(instruct_i, input_i, output_i)
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
                            max_tokens=2048,
                            top_p=1.0,
                )
                response = completion.choices[0].message.content
            except:
                response = ''

            seg_list = extract_segments(response)
            pass

        if len(seg_list) != 1:
            seg_list = ['']
    
        temp_data = {}
        temp_data['instruction'] = ori_data_i['instruction']
        temp_data['output'] = ori_data_i['output']
        temp_data['input'] = input_i
        temp_data['better_answer'] = seg_list[0]
        new_data.append(temp_data)


    if args.save_intermediate_path != '':
        with open(args.save_intermediate_path,'w') as f:
            json.dump(new_data,f,indent=4)

    final_new_data = []
    none_count = 0
    for i, data_i in enumerate(new_data):
        
        temp_data = {}
        temp_data['instruction'] = data_i['instruction']
        temp_data['input'] = data_i['input']

        if data_i['better_answer'] == '':
            none_count += 1
            temp_data['output'] = data_i['output']
        else:
            temp_data['output'] = data_i['better_answer']

    print('none_num',none_count)
    print('Len New Data', len(final_new_data))
    with open(args.save_path,'w') as f:
        json.dump(final_new_data,f,indent=4)