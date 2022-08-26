# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/eye/eye.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["n"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["n"][i], arg_data["m"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(n, m):
    return [n, m]


register_sample(__name__, get_sample_config, gen_np_args)
