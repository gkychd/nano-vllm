"""
Sampler 层实现

核心功能：
1. 将 logits 转换为概率分布
2. 使用 Gumbel-Max trick 进行采样
3. 支持温度参数控制采样随机性
"""
import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器模块

    作用：从模型输出的 logits 中采样下一个 token

    采样算法：Gumbel-Max Trick
    - 这是一种高效的离散采样方法，避免了复杂的累计分布函数计算
    - 原理：对每个 logit 加上 Gumbel 噪声，然后取 argmax
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # PyTorch 2.0 编译优化，类似 CUDA JIT
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        采样前向传播

        Args:
            logits: 模型输出的 logits，shape [batch_size, vocab_size]
            temperatures: 温度参数，shape [batch_size]
                - 温度越高，分布越平坦，采样越随机
                - 温度越低，分布越尖锐，采样越确定性

        Returns:
            sample_tokens: 采样的 token ID，shape [batch_size]

        对应 C++ 实现逻辑：
        std::vector<int> Sampler::forward(torch::Tensor logits,
                                          torch::Tensor temperatures) {
            // 1. 除以温度
            logits = logits / temperatures.unsqueeze(1)

            // 2. softmax 转概率
            probs = softmax(logits, dim=-1)

            // 3. Gumbel-Max 采样
            std::vector<int> result;
            for (int b = 0; b < batch_size; b++) {
                double max_prob = -1;
                int max_idx = 0;
                // 生成 Gumbel 噪声并采样
                for (int v = 0; v < vocab_size; v++) {
                    double gumbel_noise = -log(-uniform_random());
                    double score = probs[b][v] / gumbel_noise;
                    if (score > max_prob) {
                        max_prob = score;
                        max_idx = v;
                    }
                }
                result.push_back(max_idx);
            }
            return result;
        }
        """
        # ===== 1. 温度缩放 =====
        # logits = logits / temperature
        # unsqueeze(dim=1) 将 [batch_size] 变为 [batch_size, 1]，便于广播
        # 温度参数作用：
        #   - temperature = 1.0: 保持原始分布
        #   - temperature > 1.0: 分布更平坦，增加随机性
        #   - temperature < 1.0: 分布更尖锐，趋向高概率 token
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # ===== 2. Softmax 转为概率分布 =====
        # softmax(logits, dim=-1) 对最后一个维度（vocab_size）做 softmax
        # 输出范围 [0, 1]，所有元素和为 1
        # 对应 C++: probs[i] = exp(logits[i]) / sum(exp(logits[j]))
        probs = torch.softmax(logits, dim=-1)

        # ===== 3. Gumbel-Max 采样 =====
        # 完整公式：argmax(logits + Gumbel噪声)
        # 这里使用等价形式：argmax(probs / Gumbel噪声)
        #
        # 步骤分解：
        # a) torch.empty_like(probs).exponential_(1)
        #    - 创建与 probs 同形状的未初始化 tensor
        #    - .exponential_(1) 就地填充指数分布随机数 (λ=1)
        #    - 指数分布 PDF: f(x) = λ * exp(-λx), 这里 λ=1
        #
        # b) .clamp_min_(1e-10)
        #    - 防止除零错误
        #    - 就地操作，不创建新 tensor
        #
        # c) probs.div_(...)
        #    - 用概率除以随机数
        #    - 高概率 token 除以小随机数 → 分数更高
        #    - 低概率 token 除以大随机数 → 分数更低
        #
        # d) .argmax(dim=-1)
        #    - 返回最大分数对应的索引
        #
        # 为什么这样采样？
        # - 直接从分布采样需要计算 CDF，效率低
        # - Gumbel-Max 可以用简单的 argmax 实现采样
        # - 比 torch.multinomial 更高效（尤其在大 vocab 场景）
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
