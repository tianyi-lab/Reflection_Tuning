import json
from tqdm import tqdm

json_path_ori = ''
json_path_Q = ''
json_path_A = ''
json_path_QA = ''
save_path = ''

with open(json_path_ori,'r') as f:
    data_ori = json.load(f)
with open(json_path_Q,'r') as f:
    data_Q = json.load(f)
with open(json_path_A,'r') as f:
    data_A = json.load(f)
with open(json_path_QA,'r') as f:
    data_QA = json.load(f)

new_data = []
count_1,count_2,count_3,count_4 = 0,0,0,0
for i in tqdm(range(len(data_ori))):
    data_i_ori = data_ori[i]
    ori_ifd = data_i_ori['ifd_ppl']
    ori_rifd = data_i_ori['rifd_ppl']

    data_i_Q = data_Q[i]
    Q_ifd = data_i_Q['ifd_ppl']
    Q_rifd = data_i_Q['rifd_ppl']

    data_i_A = data_A[i]
    A_ifd = data_i_A['ifd_ppl']
    A_rifd = data_i_A['rifd_ppl']

    data_i_QA = data_QA[i]
    QA_ifd = data_i_QA['ifd_ppl']
    QA_rifd = data_i_QA['rifd_ppl']

    # check reflection on Q 
    if Q_ifd > ori_ifd: # instruction improved, thus Q* can be kept
        # QA or Q will be selected
        if QA_rifd < Q_rifd: # easier to guess, A is better, thus A* can be kept
            if QA_rifd < 1 and QA_ifd < 1:
                count_1 += 1
                new_data.append(data_i_QA)
        else: # harder to guess, A is worse, thus A* is discarded
            if Q_rifd < 1 and Q_ifd < 1:
                count_2 += 1
                # new_data.append(data_i_Q)

    else: # instruction not improved, thus Q* is discarded
        # A or ori will be selected
        if A_rifd < ori_rifd: # easier to guess, A is better, thus A* can be kept
            if A_rifd < 1 and A_ifd < 1:
                count_3 += 1
                new_data.append(data_i_A)
        else: # harder to guess, A is worse, thus A* is discarded
            if ori_rifd < 1 and ori_ifd < 1:
                count_4 += 1
                # new_data.append(data_i_ori)
    pass

print(count_1)
print(count_2)
print(count_3)
print(count_4)

print(len(new_data))
with open(save_path, 'w') as file:
    json.dump(new_data, file, indent=4)
pass