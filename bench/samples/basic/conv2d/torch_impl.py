# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer
import pandas as pd
import xlsxwriter

excel_path = "res.xlsx"

def conv2d(op, input_torch):
    ret = op(input_torch)
    ret.backward(torch.ones_like(ret).cuda())
    return ret


def args_adaptor(np_args):
    in_channels = np_args[0]
    out_channels = np_args[1]
    kernel_size = np_args[2]
    bias = np_args[3]
    stride = np_args[4]
    padding = np_args[5]
    dilation = np_args[6]
    groups = np_args[7]
    input_torch = torch.from_numpy(np_args[8]).to(torch.float32).cuda()
    conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).cuda()
    
    return [conv2d, input_torch]


def executer_creator():
    return Executer(conv2d, args_adaptor)



def bwd_get(profiler):
    res_val = 0
    key_averages = profiler.key_averages()
    specific_operators = [e for e in key_averages if "convolution_backward" in e.key]
    reduce_kernel_val = [e for e in key_averages if "reduce_kernel" in e.key]
    init_val = [e for e in key_averages if "init_device_workspace_kernel" in e.key]
    reduce_val = [e for e in key_averages if "reduce_wgrad_nchw_helper" in e.key]
    # if len(specific_operators) == 1 and len(reduce_kernel_val) == 1:
    if len(specific_operators) == 1:
        res_val = specific_operators[0].self_device_time_total
        for val in reduce_kernel_val:
            res_val -= val.self_device_time_total
        for val in init_val:
            res_val -= val.self_device_time_total
        for val in reduce_val:
            res_val -= val.self_device_time_total
        # if len(init_val) > 1:
        #     print(f"init val num large than 1, num : {len(init_val)}, id: {index}")
        # if len(reduce_val) > 1:
        #     print(f"reduce_helper val num large than 1, num : {len(reduce_val)}, id: {index}")
    else:
        print("bwd abnormal")
    return res_val

def fwd_get(profiler):
    res_val = 0
    key_averages = profiler.key_averages()
    specific_operators = [e for e in key_averages if "cudnn_convolution" in e.key]
    if len(specific_operators) == 1:
        res_val = specific_operators[0].self_device_time_total
    else:
        print("fwd abnormal")
    return res_val

def selfop(profiler):
    
    bwd_val = bwd_get(profiler)
    fwd_val = fwd_get(profiler)
    df = pd.read_excel(excel_path)
    new_row = pd.DataFrame({'fwd':[fwd_val], 'bwd':[bwd_val]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_path, index=False)



# def executer_creator():
#     return Executer(conv2d, args_adaptor)


def executer_creator():
    # 创建一个新的工作簿
    workbook = xlsxwriter.Workbook(excel_path)

    # 关闭工作簿
    workbook.close()
    return Executer(conv2d, args_adaptor,selfop)