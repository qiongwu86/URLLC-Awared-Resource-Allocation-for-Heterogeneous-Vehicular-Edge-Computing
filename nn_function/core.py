import abc

import numpy as np
import torch
from torch import nn as nn

from nn_function import pytorch_util as ptu


class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass


def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    return elem_or_tuple_to_numpy(outputs)


def torch_ify(np_array_or_other):                   # np_array_or_other = {ndarray:(1,17)}
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)    # 变成张量
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def _elem_or_tuple_to_variable(elem_or_tuple):  # elem_or_tuple(ndarray): [[-1.44583593e-01  5.81581191e-02 -3.
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple # 不懂
        )
    return ptu.from_numpy(elem_or_tuple).float()    # 转换为tensor类型


def elem_or_tuple_to_numpy(elem_or_tuple):              # elem_or_tuple =  tensor([[ 0.9896,  0.2419, -0.3484,  0.2582, -0.8483,  0.5567]])
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)


def _filter_batch(np_batch):
    for k, v in np_batch.items():   # k : 'observations' , v : [[-1.44583593e-01  5.81581191e-02 -3.02074251e-03 ...
        if v.dtype == np.bool:
            yield k, v.astype(int)  # yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始
        else:
            yield k, v              # yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始


def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {    # _elem_or_tuple_to_variable(x) 转换为tensor类型
            k: _elem_or_tuple_to_variable(x) for k, x in _filter_batch(np_batch) if x.dtype != np.dtype('O')  #
        }
    else:
        _elem_or_tuple_to_variable(np_batch)
