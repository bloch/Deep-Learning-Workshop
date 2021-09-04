import torch


# method 1 - quantization
def quantize1(x):       # x is a tensor
    x = x * 255
    x = x.to(torch.uint8)
    return x


def dequantize1(x):
    return x.to(torch.float32) / 255


# method 2 - quantization
def quantize2(x):
    S = (torch.max(x) - torch.min(x))/255
    q = (x / S).to(torch.uint8)         # quantized, here we have to quantized S as well(extra const 32float mem)
    return q, S


def dequantize2(q, S):
    return q.to(torch.float32) * S
