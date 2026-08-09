'''
    -计算int4 int5_6 int7_8的占比
'''
import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt

import time
start_time = time.time()
# ----------------------------------------------------------
methods = ['qwt_wikitext', 'qwt_c4']

models = [
            'TinyLlama-1.1B-Chat-v1.0',
            'llama-2-7b-hf',
            'Meta-Llama-3-8B',
            'Qwen2.5-0.5B',
            'Qwen2.5-1.5B',
            'Qwen2.5-7B'
            ]
# models = ['Qwen2.5-0.5B']
methods = ['qwt_wikitext']
for method in methods:
    for model in models:
        print(f'======={model}=======')
        if model == 'TinyLlama-1.1B-Chat-v1.0':
            from saved_tensor.TinyLlama_1_1B_Chat_v1_0_qwt_wikitext.activation_key import act_keys
        elif model == 'llama-2-7b-hf':
            from saved_tensor.llama_2_7b_hf_qwt_wikitext.activation_key import act_keys
        elif model == 'Meta-Llama-3-8B':
            from saved_tensor.Meta_Llama_3_8B_qwt_wikitext.activation_key import act_keys
        elif model == 'Qwen2.5-0.5B':
            from saved_tensor.Qwen2_5_0_5B_qwt_wikitext.activation_key import act_keys
        elif model == 'Qwen2.5-1.5B':
            from saved_tensor.Qwen2_5_1_5B_qwt_wikitext.activation_key import act_keys
        elif model == 'Qwen2.5-7B':
            from saved_tensor.Qwen2_5_7B_qwt_wikitext.activation_key import act_keys

        model_rename = model.replace("-", "_").replace(".", "_")
        safetensors_path_act = f"saved_tensor/{model_rename}_{method}/activation.safetensors"
        state_dict = load_file(safetensors_path_act)
        keys = act_keys

        profile_data = True
        if profile_data:
            int4_counter = int8_counter = 0
            sub_tensor_count = 0    

            with open(f'encodeIndex_{model}.txt', 'w') as f:
                pass
            for idx, key in enumerate(keys):
                w_int8 = state_dict[key].to('cuda').squeeze()
                w_int8_org_shape = w_int8.shape
                w_int8 = w_int8.reshape(-1, 128)
                max = w_int8.amax(dim=-1, keepdim=True)
                min = w_int8.amin(dim=-1, keepdim=True)

                present_range = max - min
                int8 = present_range > 63 # clamp to int8
                int8_counter += int8.sum()
                sub_tensor_count += w_int8.size(0)
                index = int8.reshape(w_int8_org_shape[0], -1).int()
                col_ones = index.sum(dim=0)
                # '''
                with open(f'encodeIndex_{model}.txt', 'a') as f:
                    '''
                    f.write(f'>>>> layerID: {idx}\n')
                    f.write(f'{col_ones.tolist()}\n')
                    f.write(f'{sorted(col_ones.tolist())}\n')
                    for vec in index[:10]:
                        f.write(f'{vec.tolist()}\n')

                    f.write(f'{idx:>3}, {int8.sum():>8}, {w_int8.size(0):>8}, {int8.sum() / w_int8.size(0):>8}\n')
                    '''
                    f.write(f'{int8.sum() / w_int8.size(0):>.4f}\n')

            with open(f'intRatio_{method}.txt', 'a') as f:
                f.writelines(f'{model} int8 Ratio: {int8_counter / sub_tensor_count :>.4f}s\n')
    # ----------------------------------------------------------
    with open(f'intRatio_{method}.txt', 'a') as f:
        end_time = time.time()
        duration = end_time - start_time
        hour = duration // 3600
        minute = (duration % 3600) // 60
        second = duration % 60
        f.writelines(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n\n')
                                




