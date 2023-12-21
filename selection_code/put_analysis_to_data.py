import json
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    if args.pt_data_path[-6:] == '.jsonl':
        pt_data = []
        with open(args.pt_data_path, 'r') as file:
            for line in file:
                pt_data.append(json.loads(line.strip()))

    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    assert len(json_data) == len(pt_data)

    new_data = []
    for i in tqdm(range(len(pt_data))):

        json_data_i = json_data[i]

        pt_data_i = pt_data[i]
        if pt_data_i == {}:
            ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = np.nan, np.nan, np.nan, np.nan
            loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = np.nan, np.nan, np.nan, np.nan
        else:
            ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = \
                pt_data_i['ppl'][0], pt_data_i['ppl'][1], pt_data_i['ppl'][2], pt_data_i['ppl'][3]
            loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = \
                pt_data_i['loss'][0], pt_data_i['loss'][1], pt_data_i['loss'][2], pt_data_i['loss'][3]

        json_data_i['ppl_Q_direct'] = ppl_Q_direct
        json_data_i['ppl_A_direct'] = ppl_A_direct
        json_data_i['ppl_Q_condition'] = ppl_Q_condition
        json_data_i['ppl_A_condition'] = ppl_A_condition
        try:
            json_data_i['ifd_ppl'] = ppl_A_condition/ppl_A_direct
            json_data_i['rifd_ppl'] = ppl_Q_condition/ppl_Q_direct
        except ZeroDivisionError:
            json_data_i['ifd_ppl'] = 0
            json_data_i['rifd_ppl'] = 0

        json_data_i['loss_Q_direct'] = loss_Q_direct
        json_data_i['loss_A_direct'] = loss_A_direct
        json_data_i['loss_Q_condition'] = loss_Q_condition
        json_data_i['loss_A_condition'] = loss_A_condition
        try:
            json_data_i['ifd_loss'] = loss_A_condition/loss_A_direct
            json_data_i['rifd_loss'] = loss_Q_condition/loss_Q_direct
        except ZeroDivisionError:
            json_data_i['ifd_loss'] = 0
            json_data_i['rifd_loss'] = 0

        new_data.append(json_data_i)

    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)


if __name__ == '__main__':
    main()