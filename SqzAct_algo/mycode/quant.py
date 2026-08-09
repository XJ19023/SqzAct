from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor, nn
import torch
import math
import init
from functools import partial
from globalVar import (get_iterationCounter,
                       get_save_tensor_enable,
                       append_activation,
                       append_weight,
                       get_data_type,
                       get_clamp_block_size,
                       get_profiling_enable)
@torch.no_grad()
def pseudo_quantize_tensor( w, 
                            n_bit=8, 
                            zero_point=True, 
                            q_group_size=-128, 
                            inplace=False, 
                            get_scale_zp=False, 
                            clam_quant_en=False, 
                            name=None,
                            spark_quant_en=False,
                            ):

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    # assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=-1, keepdim=True)
        min_val = w.amin(dim=-1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w_int = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) # quantized INT8
        if get_save_tensor_enable() and 'act' in name:
            append_activation(f'{name}_{get_iterationCounter()}', w_int)
        if get_save_tensor_enable() and 'wgt' in name:
            append_weight(f'{name}_{get_iterationCounter()}', w_int)
        if clam_quant_en:
            if q_group_size < 0: # per channel
                clamp_block_size = get_clamp_block_size()
                w_int = w_int.reshape(-1, clamp_block_size) # clamp block size
                max = w_int.amax(dim=-1, keepdim=True)
                min = w_int.amin(dim=-1, keepdim=True)
                present_range = max - min
                even = (max + min) // 2 # stored tensor
                w_int -= even
                clamp_idx = (present_range <= 63) * (present_range > 15)
            if q_group_size > 0:
                mean_bit_width = (torch.floor(torch.log2(w_int.clamp(min=1))) + 1).sum(dim=1, keepdim=True) / 128
                # mean_bit_width = torch.floor(torch.log2((tensor.sum(dim=1) / 128).clamp(min=1)).float()) + 1
                clamp_idx = (mean_bit_width <= 6) * (mean_bit_width > 4)
                even = 0
                if get_profiling_enable():
                    with open('log/profiling.txt', 'a') as f:
                        total_num = w_int.size(0)
                        int4 = mean_bit_width <= 4
                        int5_6 = (mean_bit_width > 4) * (mean_bit_width <= 6) 
                        int7_8 = mean_bit_width > 6
                        f.writelines(f'int4: {int4.sum() / total_num:>6.5f} ')
                        f.writelines(f'int5_6: {int5_6.sum() / total_num:>6.5f} ')
                        f.writelines(f'int7_8: {int7_8.sum() / total_num:>6.5f}\n')
            
            clamp_idx = clamp_idx.expand(-1, w_int.size(-1))
            w_int_clamp = w_int.masked_fill(~clamp_idx, 0)
            w_int_left = w_int.masked_fill(clamp_idx, 0)
            w_int_clamp = (w_int_clamp // 4)  # round to floor
            w_int_clamp *= 4
            w_int = w_int_clamp + w_int_left + even
            if q_group_size < 0:
                w_int = w_int.reshape(org_w_shape)
            w = (w_int- zeros) * scales
        if spark_quant_en:
            w_int = w_int.int()
            b0_1 = w_int>>7 & 1
            b3_1 = w_int>>4 & 1
            left = (1 - b0_1 - b3_1).bool()
            index_01 = ((1 - b0_1) & b3_1).bool()
            index_10 = (b0_1 & (1 - b3_1)).bool()

            tensor_01 = (w_int.masked_fill(~index_01, 0) | (index_01 * 15)) & (239)
            tensor_10 = (w_int.masked_fill(~index_10, 0) & 240) | (index_10 * 16)
            tensor_left = w_int.masked_fill(~left, 0)
            w_int_spark = tensor_01 + tensor_10 + tensor_left
            # Compute mean squared error between w_int and w_int for debugging or analysis
            # print(f'{list(w_int)}\n{list(w_int_spark)}\n')
            # print(f'mse >>> {torch.nn.functional.mse_loss(w_int.float(), w_int_spark.float()).item():.2f}')
            print(w_int - w_int_spark)
            w = (w_int_spark - zeros) * scales
        else:
            w = (w_int - zeros) * scales

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

class quantLinear(torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 name=None,
                 wgt_nbit=4,
                 act_nbit=8,
                 ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        self.quant_en = False
        self.clamp_en = False
        self.spark_en = False

        self.act_quant = partial(pseudo_quantize_tensor, n_bit=act_nbit, name=f'{name}.act')
        self.wgt_quant = partial(pseudo_quantize_tensor, n_bit=wgt_nbit, q_group_size=-1, name=f'{name}.wgt') # group-wise quant for weight

        super().__init__(in_features, out_features, bias)

    @staticmethod
    def set_param(module, name, wgt_nbit=4, act_nbit=8
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = quantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            name=name,
            wgt_nbit=wgt_nbit,
            act_nbit=act_nbit,
        )
        new_module.weight = torch.nn.Parameter(module.weight.data.clone())
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quant_en={self.quant_en}, clamp_quant={self.clamp_en}'

    def enable_quant(self, quant_en=True, clamp_en=True, spark_en=False):
        self.quant_en = quant_en
        self.clamp_en = clamp_en
        self.spark_en = spark_en

    @torch.no_grad()
    def forward(self, x):
        if self.quant_en:
            q_x = self.act_quant(x, clam_quant_en=self.clamp_en, spark_quant_en=self.spark_en)
            q_w = self.wgt_quant(self.weight)
            y = torch.functional.F.linear(q_x, q_w, self.bias)
        else:
            y = torch.functional.F.linear(x, self.weight, self.bias)
        # with open('log/qwt_bench.py', 'a') as f:
        #     f.writelines(f'[{list(x.squeeze().shape)}, {list(self.weight.shape)}, {list(y.squeeze().shape)}, [], [], 8, 1 ],\n')
        return y
