# Reflection-Tuning

This is the repo for the Reflection-Tuning project, which introduces a reflection-based method to improve the quality of instruction-tuning data.

The repo contains:

- The recycled data by our method. 
- The model checkpoints that were trained using our data.
- The code for recycling the data from the existing instruction-tuning dataset.


## News
- [2023/10] We released codes for this project.

## Overview

We propose a reflection-based method for improving the quality of instruction-response pairs. 
Given the initial base dataset, we are motivated to generate a high-quality version of each data point with an oracle model, chatGPT for instance. 
However, a common problem with using LLMs as judges is the failure to obtain diverse results. 
To overcome this potential problem, inspired by Chain-of-Thought prompting, we further define several specific criteria for the oracle model to follow, and respond to those specific criteria with critical responses, respectively. 
Then the responses to these criteria can serve as bridges (chain of thought) to generate new instruction-response pairs that are satisfied. 

## Install

Install the dependencies with `pip install -r requirements.txt`

## Run Code

### Reflection on Instruction
1. Reflection
```
python reflecn_instruction.py \
    --data_path data/alpaca_data.json \
    --save_path alpaca_reflected_instruction.json \
    --api_key xxx 
```
```--data_path```: The targeted dataset in the Alpaca format <br>
```--save_path```: The path to save the raw reflection texts <br>
```--api_key```: Your openAI key

2. Extract the instruction-response pairs:
```
python reflect_instruction_postprocess.py \
    --raw_data_path alpaca_reflected_instruction.json \
    --ori_data_path data/alpaca_data.json \
    --save_path alpaca_ins_process.json \
    --save_intermediate_path alpaca_ins_mid.json \
    --api_key xxx 
```
```--raw_data_path```: The path that saves the raw reflection texts <br>
```--ori_data_path```: The original targeted dataset in the Alpaca format <br>
```--save_path```: The path to save formated dataset in the Alpaca format <br>
```--save_intermediate_path```: The path to save the middle results <br>
```--api_key```: Your openAI key

### Reflection on Response
1. Reflection
```
python reflect_response.py \
    --data_path data/alpaca_data.json \
    --save_path alpaca_reflected_response.json \
    --api_key xxx 
```

2. Extract the instruction-response pairs:
```
python reflect_response_postprocess.py \
    --raw_data_path alpaca_reflected_response.json \
    --ori_data_path data/alpaca_data.json \
    --save_path alpaca_res_process.json \
    --save_intermediate_path alpaca_res_mid.json \
    --api_key xxx 
```

Note: Reflecting on the whole alpaca dataset will consume a lot, so we recommend using some tiny datasets for the beginning. 

## Data and Model Weights V1

The following table provides a comparison between our recycled models and baseline models on the Huggingface Open LLM Leaderboard and AlpacaEval Leaderboard. <br>
The prompt and training hyperparameters can be found in the Hyperparameters section. 
These results verify the effectiveness of our method, which can be used to improve the data samples for instruction tuning. <br>


|                          | **Avg** | **ARC** | **HellaSwag** | **MMLU** | **TruthfulQA** || **AlpacaEval** ||**Data**| **Model**|
|--------------------------|:-----------:|:-------:|:-------------:|:-------:|:--------------:|:-:|:--------------:|:-:|:-:|:-:|
| **Alpaca 7B**      | 50.21       | 42.65   | 76.91         | 41.73   | 39.55          || 26.46          ||/|/|
| **Recycled Alpaca 7B**     | 56.18| 53.92   | 77.68         | 47.55   | 45.55          || 76.99          ||[Link]|[Link]
||||||||||||
| **WizardLM 7B**    | 54.18       | 51.60   | 77.70         | 42.70   | 44.70          || 67.64          ||/|/|
| **Recycled WizardLM 7B**  | 56.21       | 53.92   | 77.05         | 48.35   | 45.52         || 78.88          ||[Link]|[Link]
||||||||||

## Prompt and Hyperparameters

We use the prompt from [FastChat](https://github.com/lm-sys/FastChat):

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi ASSISTANT: Hello.</s>USER: Who are you? ASSISTANT: I am ...</s>......
```

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay | Warmup Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recycled Models (7B) | 128 | 2e-5 | 3 | 2048 | 0 | 0.03 |
| Recycled Models (13B) | 128 | 2e-5 | 3 | 2048 | 0 | 0.03 |

## ToDo
- [ ] Release the code, data, and models. 
- [ ] Train 13B models.
- [ ] Release new versions.
