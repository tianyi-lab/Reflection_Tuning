# Reflection-Tuning: An Approach for Data Recycling

[Paper V1] [Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning](https://arxiv.org/abs/2310.11716)

<p align="center" width="40%">
<a ><img src="images/alpaca1.jpg" alt="overview" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

This is the repo for the Reflection-Tuning project, which introduces a reflection-based method to improve the quality of instruction-tuning data.

The repo contains:

- The recycled data by our method. 
- The model checkpoints that were trained using our data.
- The code for recycling the data from the existing instruction-tuning dataset.

## News
- [2023/10] We released the V2 model and pushed models to the Alpaca Eval leaderboard. 
- [2023/10] We released codes for this project.

## Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Install](#install)
- [Run Code](#run-code)
- [Data and Model Weights V1](#data-and-model-weights-v1)
- [Data and Model Weights V2](#data-and-model-weights-v2)
- [Prompt and Hyperparameters](#prompt-and-hyperparameters)
- [ToDo](#todo)
- [Citation](#citation)


## Overview

We propose a reflection-based method for improving the quality of instruction-response pairs. 
Given the initial base dataset, we are motivated to generate a high-quality version of each data point with an oracle model, chatGPT for instance. 
However, a common problem with using LLMs as judges is the failure to obtain diverse results. 
To overcome this potential problem, inspired by Chain-of-Thought prompting, we further define several specific criteria for the oracle model to follow, and respond to those specific criteria with critical responses, respectively. 
Then the responses to these criteria can serve as bridges (chain of thought) to generate new instruction-response pairs that are satisfied. 

## Highlights

* In Reflection-Tuning V1, we propose a reflection method that can improve the quality of the instruction-tuning dataset, which is a general method and can be utilized on almost ANY instruction-tuning dataset.
* We implement our method on both [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [WizardLM](https://github.com/nlpxucan/WizardLM) datasets and release the newly-generated high-quality recycled datasets. 

## Install

Install the dependencies with `pip install -r requirements.txt`

## Run Code
#### Note: 
Reflecting on the whole dataset containing dozens of thousands of data will consume a lot, so we recommend using some tiny datasets for the beginning, for example, cherry data from [Cherry LLM](https://github.com/MingLiiii/Cherry_LLM). Experiments show that simply reflecting on a subset of high-quality data can also get a promising performance. <br>
In the below scripts, we directly run on ```data/cherry_alpaca_v1/cherry_alpaca_5_percent_compressed.json``` which contains only approximately 3k [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data. 

### Reflection on Instruction


1. Reflection
```
python reflection_code/reflecn_instruction.py \
    --data_path data/cherry_alpaca_v1/cherry_alpaca_5_percent_compressed.json \
    --save_path cherry_alpaca_5_percent_reflect_ins_raw.json \
    --api_key xxx 
```
```--data_path```: The targeted dataset in the Alpaca format <br>
```--save_path```: The path to save the raw reflection texts <br>
```--api_key```: Your openAI key

2. Extract the instruction-response pairs:
```
python reflection_code/reflect_instruction_postprocess.py \
    --raw_data_path cherry_alpaca_5_percent_reflect_ins_raw.json \
    --ori_data_path data/cherry_alpaca_v1/cherry_alpaca_5_percent_compressed.json \
    --save_path cherry_alpaca_5_percent_reflect_ins.json \
    --save_intermediate_path cherry_alpaca_5_percent_reflect_ins_mid.json \
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
python reflection_code/reflect_response.py \
    --data_path data/cherry_alpaca_v1/cherry_alpaca_5_percent_compressed.json \
    --save_path cherry_alpaca_5_percent_reflect_res_raw.json \
    --api_key xxx 
```

2. Extract the instruction-response pairs:
```
python reflection_code/reflect_response_postprocess.py \
    --raw_data_path cherry_alpaca_5_percent_reflect_res_raw.json \
    --ori_data_path data/cherry_alpaca_v1/cherry_alpaca_5_percent_compressed.json \
    --save_path cherry_alpaca_5_percent_reflect_res.json \
    --save_intermediate_path cherry_alpaca_5_percent_reflect_res_mid.json \
    --api_key xxx 
```

Note: When reflecting on the instruction, we first compress the instruction and input it into one single instruction for easier processing by using chatGPT. 
The whole compressed Alpaca data can be found in the data folder. <br>
Note: The extraction of reflection results is based on regular expression and, thus is not perfect. We will release the raw output before the extraction in the future. 

## Data and Model Weights V1

The following table provides a comparison between our recycled models (V1) and baseline models on the AlpacaEval Leaderboard and Huggingface Open LLM Leaderboard. <br>

The Recycled Alpaca Data can be found here: [[hf-Link]](https://huggingface.co/datasets/umd-zhou-lab/recycled_alpaca_v1) <br>
The Recycled WizardLM (70k) Data can be found here: [[hf-Link]](https://huggingface.co/datasets/umd-zhou-lab/recycled_wiz70_v1) <br>

|                          | **AlpacaEval** || **Avg** | **ARC** | **HellaSwag** | **MMLU** | **TruthfulQA** || **Model**|
|--------------------------|:--------------:|:-:|:-----------:|:-------:|:-------------:|:-------:|:--------------:|:-:|:-:|
| **Alpaca 7B**            | 26.46          || 50.21       | 42.65   | 76.91         | 41.73   | 39.55          ||/|
| **Recycled Alpaca 7B V1.0**   | 76.99          || 56.18| 53.92   | 77.68         | 47.55   | 45.55          ||[[hf-Link]](https://huggingface.co/umd-zhou-lab/recycled-alpaca-7b-v1.0)|
| **Recycled Alpaca 13B V1.0**  | 83.42          || 58.93| 58.70   | 80.80         | 53.11   | 43.12          ||[Link]|
|||||||||||
| **WizardLM 7B**          | 67.64          || 54.18       | 51.60   | 77.70         | 42.70   | 44.70          ||/|
| **Recycled WizardLM 7B V1.0** | 78.88          || 56.21       | 53.92   | 77.05         | 48.35   | 45.52         ||[[hf-Link]](https://huggingface.co/umd-zhou-lab/recycled-wizardlm-7b-v1.0)|
|||||||||

## Data and Model Weights V2

The following table provides a comparison between our recycled models (V2) and baseline models on the AlpacaEval Leaderboard and Huggingface Open LLM Leaderboard. <br>

The V2 Recycled Alpaca Data and WizardLM data, and the corresponding paper will be released soon. 

|                          | **AlpacaEval** || **Avg** | **ARC** | **HellaSwag** | **MMLU** | **TruthfulQA** || **Model**|
|--------------------------|:--------------:|:-:|:-----------:|:-------:|:-------------:|:-------:|:--------------:|:-:|:-:|
| **Alpaca 7B**            | 26.46          || 50.21       | 42.65   | 76.91         | 41.73   | 39.55          ||/|
| **Recycled Alpaca 7B V2.0**   |     79.58      || 56.05 |  54.01  |   78.07      |  46.69  |    45.41       |||
|||||||||||
| **WizardLM 7B**          | 67.64          || 54.18       | 51.60   | 77.70         | 42.70   | 44.70          ||/|
| **Recycled WizardLM 7B V2.0** |     83.48      ||   56.79     |  54.78 |     77.86     |  45.63  |    48.91     |||
|||||||||

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
- [x] Release the code, data, and models.
- [ ] Train and release V1 13B models.
- [ ] Release the data and paper for V2 models. 
- [ ] Train and release V2 13B models.

## Citation

Please consider citing our paper if you think our codes, data, or models are useful. Thank you!
```
@misc{li2023reflectiontuning,
      title={Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning}, 
      author={Ming Li and Lichang Chen and Jiuhai Chen and Shwai He and Heng Huang and Jiuxiang Gu and Tianyi Zhou},
      year={2023},
      eprint={2310.11716},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
