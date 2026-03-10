"""
模型权重加载模块

功能：
1. 从 HuggingFace 格式的 checkpoint 加载权重
2. 支持 safetensors 格式
3. 支持合并拆分模块（QKV 合并、Gate+Up 合并）
"""
import os
from glob import glob  # 文件路径匹配
import torch
from torch import nn
from safetensors import safe_open  # 安全的 tensor 加载


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载器

    Args:
        param: 目标参数
        loaded_weight: 要加载的权重

    直接复制数据
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重

    支持：
    1. 普通权重：直接加载
    2. 合并权重（packed_modules）：拆分后加载到对应分片

    示例（Qwen3 模型）：
    - 原始 HF 权重：q_proj, k_proj, v_proj（三个独立矩阵）
    - 合并后：qkv_proj（一个矩阵）

    权重映射：
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    加载过程：
    1. 读取 q_proj 权重
    2. 查找映射：q_proj → qkv_proj
    3. 调用 qkv_proj 的 weight_loader，传入 shard_id="q"
    4. weight_loader 自动将 q_proj 放到 qkv_proj 的正确位置

    Args:
        model: 要加载权重的模型
        path: 模型权重目录路径
    """
    # 获取模型的 packed_modules_mapping
    # 例如：{"q_proj": ("qkv_proj", "q"), ...}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历目录下所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # 打开 safetensors 文件
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重
            for weight_name in f.keys():
                # 检查是否是合并模块（packed module）
                matched = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 找到匹配的映射
                        # 例如：weight_name="model.layers.0.self_attn.q_proj"
                        # k="q_proj" → v="qkv_proj", shard_id="q"
                        v, shard_id = packed_modules_mapping[k]

                        # 替换权重名
                        # q_proj → qkv_proj
                        param_name = weight_name.replace(k, v)

                        # 获取目标参数
                        param = model.get_parameter(param_name)

                        # 获取自定义的 weight_loader
                        weight_loader = getattr(param, "weight_loader")

                        # 调用 weight_loader 加载权重
                        # 传入原始权重名和分片 ID
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        matched = True
                        break

                # 如果不是合并模块，使用默认加载方式
                if not matched:
                    param = model.get_parameter(weight_name)
                    # 获取 weight_loader（如果有的话）
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
