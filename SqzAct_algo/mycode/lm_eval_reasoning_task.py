
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          )
from tqdm import tqdm
import torch.distributed as dist

from datasets import load_dataset
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from datetime import datetime
import sys
from quant import quantLinear
from globalVar import (increas_iterationCounter,
                       save_tensors,
                       set_save_tensor_enable,
                       set_data_type,
                       set_clamp_block_size,
                       get_input_id,
                       set_profiling_enable)

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

def gather_tensor_from_multi_processes(input, world_size):
    if world_size == 1:
        return input
    torch.cuda.synchronize()
    dist.all_gather(gathered_tensors, input)
    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    torch.cuda.synchronize()

    return gathered_tensors
def lienar_regression(X, Y, block_id=0):
    # print(X.shape, Y.shape)
    gpu_num = torch.cuda.device_count()
    X = X.to(f'cuda:{gpu_num-1}')
    Y = Y.to(f'cuda:{gpu_num-1}')
    X = X.reshape(-1, X.size(-1)).float()

    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1)).float()

    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)

    # _write('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    # W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y
    W_overall = torch.linalg.solve(X_add_one_T @ X_add_one, X_add_one_T @ Y)

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    # Y_pred = X @ W + b
    batch_size = 1024  # 可以根据显存大小调整
    if gpu_num == 2:
        Y_pred = torch.empty((X.size(0), W.size(1)), dtype=torch.bfloat16, device="cuda:1")
    else:
        Y_pred = torch.empty((X.size(0), W.size(1)), dtype=torch.bfloat16, device="cuda")
        Y = Y.to("cuda")
    for i in range(0, X.size(0), batch_size):
        X_chunk = X[i:i+batch_size]   # 取一部分样本
        Y_pred_chunk = X_chunk @ W + b
        Y_pred[i:i+batch_size] = Y_pred_chunk
    del X
    # abs_loss = (Y - Y_pred).abs().mean()

    # ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    # ss_res = torch.sum((Y - Y_pred).pow(2))
    # r2_score = 1 - ss_res / ss_tot

    ss_tot, ss_res = 0.0, 0.0
    Y_mean = Y.mean(dim=0, keepdim=True)
    # batch_size = 1024
    for i in range(0, len(Y), batch_size):
        Y_b = Y[i:i+batch_size]
        Y_pred_b = Y_pred[i:i+batch_size]
        ss_tot += torch.sum((Y_b - Y_mean) ** 2).item()
        ss_res += torch.sum((Y_b - Y_pred_b) ** 2).item()

    r2_score = 1 - ss_res / ss_tot


    # _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))
    del Y
    torch.cuda.empty_cache()
    return W, b, r2_score
def tensor_gpu_memory(tensor, name=None):
    if tensor.is_cuda:
        size_in_bytes = tensor.element_size() * tensor.numel()
        size_in_MB = size_in_bytes / 1024**2
        print(f"{name if name else ''} size: {size_in_MB:.2f} MB")
    else:
        print(f"{name if name else ''} is not on GPU.")
# current date and time
current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
start_time = time.time()
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_name", type=str, default="LLaMA-2-7B")
parser.add_argument("--task", type=str, default="wikitext")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--clamp_block_size", type=int, default=64)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument('--start_block', default=0, type=int)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--wgt_nbit", default=4, type=int)
parser.add_argument("--act_nbit", default=8, type=int)
parser.add_argument("--num_epochs", default=3, type=int)
parser.add_argument("--eval_base", action="store_true")
parser.add_argument("--eval_quant", action="store_true")
parser.add_argument("--eval_clamp", action="store_true")
parser.add_argument("--eval_quant_qwt", action="store_true")
parser.add_argument("--eval_clamp_qwt", action="store_true")
parser.add_argument("--profiling", action="store_true")
parser.add_argument("--save_tensor", action="store_true")
args = parser.parse_args()

@torch.no_grad()
def evaluate(model, dataset, n_samples=None):
    model.eval()
    nlls = []
    length = 2048
    n_samples = n_samples if n_samples else dataset.size(1) // length
    for i in tqdm(range(n_samples), desc="Evaluating..."):
        batch = dataset[:, (i * length) : ((i + 1) * length)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = dataset[:, (i * length) : ((i + 1) * length)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * length
        nlls.append(neg_log_likelihood)

        _ = increas_iterationCounter()

    return torch.exp(torch.stack(nlls).sum() / (n_samples * length))

args.world_size = 1
args.rank = 0  # global rank
alpha = args.alpha
model_path = '/data1/juxin/models/' + args.model_name
act_scales_path = args.act_scales_path
n_samples = args.n_samples
train_samples = 64 # 64
set_clamp_block_size(args.clamp_block_size)


tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_wikitext2(tokenizer, eval=True):
    if eval:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids
    return testenc

def get_c4(seqlen, tokenizer, eval=False):
    if eval:
        valdata = load_dataset(
            "json",
            data_files={"validation": "/cephfs/shared/juxin/dataset/c4/en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    else:
        valdata = load_dataset(
            "json",
            data_files={"train": "/cephfs/shared/juxin/dataset/c4/en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] == seqlen:
            # rare case, discovered with Yi tokenizer
            valenc.append(tmp.input_ids)
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc    


if args.task == 'wikitext':
    test_data = get_wikitext2(tokenizer, True)
    train_data = get_wikitext2(tokenizer, False)
if args.task == 'c4':
    test_data = get_c4(2048, tokenizer, True)
    train_data = get_c4(2048, tokenizer, True)

os.makedirs(f'log_zero_shot/{args.model_name}', exist_ok=True)

if True:
    import importlib.util
    # replace modeling_bert.py
    # 1. 加载你本地的 modeling_bert.py 文件
    if 'llama' in args.model_name.lower():
        spec = importlib.util.spec_from_file_location(
            "transformers.models.llama.modeling_llama", 
            "./mycode/modeling_llama.py"
        )
        custom_llama = importlib.util.module_from_spec(spec)
        sys.modules["transformers.models.llama.modeling_llama"] = custom_llama
        spec.loader.exec_module(custom_llama)
    if 'opt' in args.model_name.lower():
        spec = importlib.util.spec_from_file_location(
            "transformers.models.opt.modeling_opt", 
            "./mycode/modeling_opt.py"
        )
        custom_opt = importlib.util.module_from_spec(spec)
        sys.modules["transformers.models.opt.modeling_opt"] = custom_opt
        spec.loader.exec_module(custom_opt)
    if 'qwen3' in args.model_name.lower():
        spec = importlib.util.spec_from_file_location(
            "transformers.models.qwen3.modeling_qwen3", 
            "./mycode/modeling_qwen3.py"
        )
        custom_opt = importlib.util.module_from_spec(spec)
        sys.modules["transformers.models.qwen3.modeling_qwen3"] = custom_opt
        spec.loader.exec_module(custom_opt)
    if 'qwen2' in args.model_name.lower() or 'deepseek' in args.model_name.lower():
        spec = importlib.util.spec_from_file_location(
            "transformers.models.qwen2.modeling_qwen2", 
            "./mycode/modeling_qwen2.py"
        )
        custom_opt = importlib.util.module_from_spec(spec)
        sys.modules["transformers.models.qwen2.modeling_qwen2"] = custom_opt
        spec.loader.exec_module(custom_opt)

def set_quant_state(model, quant=True, clamp=False):
    for name, module in model.named_modules():
        if isinstance(module, quantLinear):
            module.enable_quant(quant, clamp)

config = AutoConfig.from_pretrained(model_path)
config.use_cache = False  # ✅ 显式修改
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    # device_map="auto",
    config=config
)

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name != 'lm_head':
        new_layer = quantLinear.set_param(module, name=name, wgt_nbit=args.wgt_nbit, act_nbit=args.act_nbit)
        _set_module(model, name, new_layer)

#=======================================================================
@torch.no_grad()
def cal_wandb_to_full(model, task, dataset, train_samples=None, clamp=None, model_name=None, do_train=False):
    # train_samples = train_samples if train_samples else len(dataset)
    model.eval()
    total_rows = 0
    for item in dataset:
        total_rows += item.shape[1] * item.shape[0]

    if 'opt' in args.model_name.lower():
        forward_before_blocks = model.model.decoder.forward_before_blocks
        layers = model.model.decoder.layers
    if 'llama' in args.model_name.lower():
        forward_before_blocks = model.model.forward_before_blocks
        layers = model.model.layers
    if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower() or 'deepseek' in args.model_name.lower():
        forward_before_blocks = model.model.forward_before_blocks
        layers = model.model.layers

    mx_accept_mse = 10000
    hidden_dim = 4096
    layer_quant_qwt = [31]
    layer_inputs_list = []
    position_ids_list, cache_position_list, position_embeddings_list = [], [], []
    for i in tqdm(range(train_samples), desc="Before layers..."):
        batch = dataset[i].to(model.device)
        with torch.no_grad():
            if 'opt' in args.model_name.lower():
                hidden_states, causal_attention_mask, use_cache = forward_before_blocks(batch)
            # if 'llama' in args.model_name.lower():
            #     hidden_states, position_ids, past_key_values, use_cache, cache_position, position_embeddings = forward_before_blocks(batch)
            if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                hidden_states, position_ids, _, _, cache_position, position_embeddings = forward_before_blocks(batch)
                # print(i, hidden_states.shape, position_ids.shape, cache_position.shape)
            layer_inputs_list.append(hidden_states.detach())
            position_ids_list.append(position_ids.detach())
            cache_position_list.append(cache_position.detach())
            position_embeddings_list.append(position_embeddings)
            del batch, hidden_states
            torch.cuda.empty_cache()

    layer_inputs_full_list = layer_inputs_list
    for layer_idx, layer in tqdm(enumerate(layers), total=len(layers), desc="In layers"):
        # layer_outputs = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cpu")
        layer_outputs_list = []
        # layer_outputs_quant = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cpu")
        layer_outputs_quant_list = []
        for input_idx, (layer_input, layer_input_full) in enumerate(zip(layer_inputs_list, layer_inputs_full_list)):
            layer_input = layer_input.to(next(layer.parameters()).device)
            layer_input_full = layer_input_full.to(next(layer.parameters()).device)
            # layer_input.unsqueeze_(0)
            # layer_input_full.unsqueeze_(0)
            set_quant_state(layer, quant=False, clamp=False)
            # append_activation(f'qwt_layers.{layer_idx}.input_full', layer_input_full)
            if 'opt' in args.model_name.lower():
                layer_output = layer(layer_input_full, causal_attention_mask, use_cache=use_cache)
            # if 'llama' in args.model_name.lower():
            #     layer_output = layer(layer_input_full, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                # print(input_idx, layer_input_full.shape, position_ids_list[input_idx].shape, cache_position_list[input_idx].shape)
                layer_output = layer(layer_input_full, position_ids=position_ids_list[input_idx], cache_position=cache_position_list[input_idx], position_embeddings=position_embeddings_list[input_idx], flag=True)
                # print(f'layer {layer_idx} input_idx {input_idx} layer_output shape: {layer_output[0].shape}')
            layer_outputs_list.append(layer_output[0].detach())
            set_quant_state(layer, quant=True, clamp=clamp)
            if layer_idx in layer_quant_qwt:
                set_quant_state(layer, quant=True, clamp=False)
            if 'opt' in args.model_name.lower():
                layer_output_quant = layer(layer_input, causal_attention_mask, use_cache=use_cache)
            # if 'llama' in args.model_name.lower():
            #     layer_output_quant = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                layer_output_quant = layer(layer_input, position_ids=position_ids_list[input_idx], cache_position=cache_position_list[input_idx], position_embeddings=position_embeddings_list[input_idx])
            layer_outputs_quant_list.append(layer_output_quant[0].detach())

        layer_inputs = torch.empty((total_rows, hidden_dim), dtype=torch.bfloat16)
        layer_outputs = torch.empty((total_rows, hidden_dim), dtype=torch.bfloat16)
        layer_outputs_quant = torch.empty((total_rows, hidden_dim), dtype=torch.bfloat16)
        current_row = 0
        for input_idx, (layer_input, layer_output, layer_output_quant) in enumerate(zip(layer_inputs_list, layer_outputs_list, layer_outputs_quant_list)):
            # print(layer_input.shape, layer_output.shape, layer_output_quant.shape)
            layer_inputs[current_row:current_row+layer_input.shape[0]*layer_input.shape[1]] = layer_input.reshape(-1, layer_input.shape[-1])
            layer_outputs[current_row:current_row+layer_input.shape[0]*layer_input.shape[1]] = layer_output.reshape(-1, layer_input.shape[-1])
            layer_outputs_quant[current_row:current_row+layer_input.shape[0]*layer_input.shape[1]] = layer_output_quant.reshape(-1, layer_input.shape[-1])
            current_row += layer_input.shape[0] * layer_input.shape[1]
            # layer_inputs = layer_input
            # layer_outputs = layer_output
            # layer_outputs_quant = layer_output_quant

        loss = nn.MSELoss()
        quant_loss = loss(layer_outputs.float(), layer_outputs_quant.float())
        W, b, r2_score = lienar_regression(layer_inputs, layer_outputs - layer_outputs_quant, block_id=layer_idx)
        if layer_idx > 2 and r2_score > 0:
            layer.set_qwt_para(W, b, r2_score)
            set_quant_state(layer, quant=True, clamp=clamp)
            if layer_idx in layer_quant_qwt:
                set_quant_state(layer, quant=True, clamp=False)
            layer_outputs_quant_list = []
            for input_idx, layer_input in enumerate(layer_inputs_list):
                layer_input = layer_input.to(next(layer.parameters()).device)
                # layer_input.unsqueeze_(0)
                if 'opt' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                # if 'llama' in args.model_name.lower():
                #     layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids_list[input_idx], cache_position=cache_position_list[input_idx], position_embeddings=position_embeddings_list[input_idx])
                layer_outputs_quant_list.append(layer_output_wandb[0].detach())

            current_row = 0
            for input_idx, layer_output_quant in enumerate(layer_outputs_quant_list):
                layer_outputs_quant[current_row:current_row+layer_output_quant.shape[0]*layer_output_quant.shape[1]] = layer_output_quant.reshape(-1, layer_output_quant.shape[-1])
                current_row += layer_output_quant.shape[0]*layer_output_quant.shape[1]
                # layer_outputs_quant = layer_output_quant
            
            qwt_loss = loss(layer_outputs_quant.float(), layer_outputs.float())

            # if qwt_loss > mx_accept_mse:
            if qwt_loss > quant_loss:
                layer.set_qwt_para(torch.zeros_like(W), torch.zeros_like(b), 1)
                set_quant_state(layer, quant=True, clamp=False)
                layer_outputs_quant_list = []
                for input_idx, layer_input in enumerate(layer_inputs_list):
                    layer_input = layer_input.to(next(layer.parameters()).device)
                    # layer_input.unsqueeze_(0)
                    if 'opt' in args.model_name.lower():
                        layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                    # if 'llama' in args.model_name.lower():
                    #     layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                    if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                        layer_output_wandb = layer(layer_input, position_ids=position_ids_list[input_idx], cache_position=cache_position_list[input_idx], position_embeddings=position_embeddings_list[input_idx])
                    layer_outputs_quant_list.append(layer_output_wandb[0].detach())
            layer_inputs_list = layer_outputs_quant_list
        else:
         
            layer.set_qwt_para(torch.zeros_like(W), torch.zeros_like(b), 1)
            set_quant_state(layer, quant=True, clamp=False)
            layer_outputs_quant_list = []
            for input_idx, layer_input in enumerate(layer_inputs_list):
                layer_input = layer_input.to(next(layer.parameters()).device)
                # layer_input.unsqueeze_(0)
                if 'opt' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                # if 'llama' in args.model_name.lower():
                #     layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                if 'qwen' in args.model_name.lower() or 'llama' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids_list[input_idx], cache_position=cache_position_list[input_idx], position_embeddings=position_embeddings_list[input_idx])
                layer_outputs_quant_list.append(layer_output_wandb[0].detach())
            layer_inputs_list = layer_outputs_quant_list
        layer_inputs_full_list = layer_outputs_list
        model.cuda()

    del layer_inputs, layer_outputs, layer_outputs_quant
    torch.cuda.empty_cache()

    return train_samples, layer_quant_qwt
    # exit()
#=======================================================================

if args.eval_base:
    print(f'---eval {args.model_name} base---')
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
    task_names = args.task.split(",")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        num_fewshot=0,
        # limit=10,
        gen_kwargs={
                "do_sample": False,   # 必须设为 True，否则温控和 top_p 不生效
                # "temperature": 0.6,
                # "top_p": 0.95,
                "max_gen_toks": 32768 # AIME 任务建议调大生成长度，防止推理被截断
            }
    )
    result = make_table(results)
    print(result)

    with open(f'log_zero_shot/{args.model_name}/zero_shot.txt', 'a') as f:
        f.writelines(f'base\n')
        f.write(result)

elif args.eval_quant:
    print(f'---eval {args.model_name} quant---')
    set_quant_state(model, quant=True, clamp=False)
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
    # print(lm_eval_model.model)
    task_names = args.task.split(",")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        num_fewshot=0,
        # limit=100,
    )
    result = make_table(results)
    print(result)

    with open(f'log_zero_shot/{args.model_name}/zero_shot.txt', 'a') as f:
        f.writelines(f'W4A8\n')
        f.write(result)

elif args.eval_clamp:
    print(f'---eval {args.model_name} clamp---')
    set_quant_state(model, quant=True, clamp=True)
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
    task_names = args.task.split(",")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        num_fewshot=0,
        # limit=100,
    )
    result = make_table(results)
    print(result)

    with open(f'log_zero_shot/{args.model_name}/zero_shot.txt', 'a') as f:
        f.writelines(f'clamp\n')
        f.write(result)

elif args.eval_clamp_qwt:
    print(f'---eval {args.model_name} clamp qwt---')
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
    task_names = args.task.split(",")

    set_save_tensor_enable()
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        num_fewshot=0,
        # limit=20,
    )
    samples = min(80, len(get_input_id()))
    input_ids = get_input_id()[:samples]

    train_samples, layer_quant_qwt = cal_wandb_to_full(lm_eval_model.model, args.task, input_ids, len(input_ids), clamp=True, model_name=args.model_name)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        num_fewshot=0,
        # limit=20,
    )
    result = make_table(results)
    print(result)

    with open(f'log_zero_shot/{args.model_name}/zero_shot.txt', 'a') as f:
        f.writelines(f'clamp_comp\n')
        f.write(result)
