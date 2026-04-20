"""
Qwen3.5-35B-A3B REAP pruned model -> GPTQ 4-bit quantization
Designed for A100 80GB: entire model fits in GPU memory.
"""

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

# ===== 路径配置 =====
model_path = "/home/reap/artifacts/Qwen3.5-35B-A3B/composite_8f6ab4f6/pruned_models/reap-seed_42-0.40"
save_path = "/data/Qwen3.5-35B-A3B-REAP-40pct-GPTQ-4bit"

# ===== Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ===== 校准数据 =====
print("正在加载校准数据集...")
dataset = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
calibration_data = dataset.select(range(256))

def prepare_calibration(examples):
    texts = []
    for instr in examples["instruction"]:
        messages = [{"role": "user", "content": instr}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=2048, padding=True)

calibration_dataset = calibration_data.map(prepare_calibration, batched=True)

# ===== 不量化的模块 =====
modules_to_not_convert = [
    "lm_head",
    "model.embed_tokens",
    "gate",
    "shared_expert_gate",
]

# ===== 量化配置 =====
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    damp_percent=0.01,
    static_groups=False,
    sym=True,
    true_sequential=True,
    offload_to_disk=False,
    modules_to_not_convert=modules_to_not_convert,
)

# ===== 加载模型 =====
print("正在加载剪枝模型到 GPU...")
model = GPTQModel.from_pretrained(
    model_path,
    quant_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# ===== 量化 =====
print("开始 GPTQ 量化 (全 GPU)...")
model.quantize(calibration_dataset)

# ===== 保存 =====
print(f"保存到 {save_path}...")
model.save_quantized(save_path)
tokenizer.save_pretrained(save_path)

print("量化完成!")
