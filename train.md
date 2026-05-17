# `training.py` 微调流程说明

本文档说明 `training.py` 如何使用训练数据对基础语言模型进行 LoRA/QLoRA 风格的监督微调。

## 1. 训练输入

`training.py` 通过命令行参数接收训练所需配置：

```bash
python training.py \
  --train_data_file "data/train_with_token.json" \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "checkpoints/llama2_lora"
```

主要参数含义：

- `--train_data_file`：训练数据路径，通常是 `create_data_split.py` 生成的 `data/train_with_token.json`。
- `--model_name`：Hugging Face 上的基础 causal language model 名称。
- `--output_dir`：LoRA adapter 和训练状态保存目录。
- `--cache_dir`：可选参数，用于指定模型下载缓存目录。

如果模型需要 Hugging Face 访问权限，脚本会从环境变量中读取：

```python
hf_token = os.getenv("HF_ACCESS_TOKEN")
```

因此 token 应配置在本地环境变量中，不应写入源码。

## 2. 读取训练数据

训练数据由 `datasets.load_dataset` 读取：

```python
dataset = load_dataset(
    "json",
    data_files=training_data_file,
    split="train",
)
```

这里的训练文件来自 `create_data_split.py`。每条样本包含：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "token": "..."
}
```

其中：

- `system`：固定自动驾驶轨迹规划任务说明。
- `user`：由原始场景数据转换出的感知、预测、ego 状态、历史轨迹和任务目标。
- `assistant`：监督答案，包括 Thoughts、Meta Action 和未来 6 个轨迹点。

## 3. 构造 4-bit 量化配置

脚本通过 `create_bnb_config()` 创建 BitsAndBytes 量化配置：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

这相当于 QLoRA 风格的加载方式：

- `load_in_4bit=True`：用 4-bit 方式加载基础模型权重，降低显存占用。
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化格式，适合神经网络权重分布。
- `bnb_4bit_use_double_quant=True`：对量化常数再次量化，进一步节省显存。
- `bnb_4bit_compute_dtype=torch.bfloat16`：量化计算时使用 bfloat16。

基础模型主体会以量化形式加载，后续训练主要更新新增的 LoRA adapter 参数。

## 4. 加载基础模型和 tokenizer

`load_model()` 使用 Hugging Face Transformers 加载模型：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={i: max_memory for i in range(n_gpus)},
    cache_dir=cache_dir,
    token=security_token,
)
```

关键点：

- `AutoModelForCausalLM` 表示训练目标是 causal language modeling，即根据前文预测后续 token。
- `quantization_config=bnb_config` 表示以 4-bit 量化方式加载模型。
- `device_map="auto"` 表示自动把模型放到可用 GPU 上。
- `max_memory` 为每张 GPU 设置显存上限。

随后加载 tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.pad_token = tokenizer.eos_token
```

部分 LLaMA 类模型默认没有 `pad_token`。训练 batch 需要 padding，因此脚本把 `eos_token` 复用为 `pad_token`。

## 5. 获取模型最大上下文长度

`get_max_length(model)` 会从模型配置中查找最大序列长度：

```python
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
    max_length = getattr(model.config, length_setting, None)
```

不同模型保存上下文长度的字段名不完全一致，因此脚本依次检查常见字段。如果没有找到，则默认使用：

```python
max_length = 1024
```

该长度会用于后续 tokenizer 截断和样本过滤。

## 6. 训练样本预处理

预处理入口是：

```python
dataset = preprocess_dataset(tokenizer, max_length, 0, dataset)
```

它包含四个步骤。

### 6.1 把 Chat messages 拼接为 text

`create_prompt_formats()` 会把每条样本的 `messages` 转为一个纯文本字段 `text`。

训练时拼接内容包括：

```text
system
<system content>

user
<user content>

assistant
<assistant content>

### End
```

代码中对应逻辑为：

```python
system = f"{message[0]['role']}\n{message[0]['content']}"
user = f"{message[1]['role']}\n{message[1]['content']}"
assistant = f"{message[2]['role']}\n{message[2]['content']}"
parts = [system, user, assistant, end]
sample["text"] = "\n\n".join(parts)
```

这样做的目的，是把原始 Chat 格式训练样本转换成 causal LM 可以直接学习的连续文本。模型训练时会看到完整的 system、user 和 assistant，并学习整段文本的下一个 token 预测。

### 6.2 Tokenize 文本

`preprocess_batch()` 调用 tokenizer：

```python
tokenizer(
    batch["text"],
    max_length=max_length,
    truncation=True,
)
```

输出会包含 `input_ids`、`attention_mask` 等字段。超过 `max_length` 的文本会被截断，避免序列过长导致显存占用过高。

### 6.3 过滤过长样本

tokenize 后，脚本继续过滤长度达到或超过 `max_length` 的样本：

```python
dataset = dataset.filter(
    lambda sample: len(sample["input_ids"]) < max_length
)
```

这样保留下来的样本都能完整落在模型上下文窗口中。

### 6.4 打乱训练集

最后使用固定 seed 打乱数据：

```python
dataset = dataset.shuffle(seed=seed)
```

这可以降低样本原始顺序对训练的影响。

## 7. 准备量化模型训练

进入 `train()` 后，脚本先开启梯度检查点：

```python
model.gradient_checkpointing_enable()
```

梯度检查点会减少训练时保存的中间激活，从而降低显存占用，但会增加一定计算开销。

随后调用 PEFT 的工具函数：

```python
model = prepare_model_for_kbit_training(model)
```

该函数会对 k-bit 量化模型做训练前准备，例如处理需要梯度的输入、layer norm 精度和其他与量化训练相关的细节。

## 8. 自动查找 LoRA 注入层

`find_all_linear_names(model)` 会遍历模型所有模块，找出类型为：

```python
bnb.nn.Linear4bit
```

的线性层名称。

这些层是 4-bit 量化后的线性层，也是 LoRA adapter 的注入目标。脚本会把扫描到的模块名收集为 `target_modules`。

如果包含 `lm_head`，脚本会移除它：

```python
if "lm_head" in lora_module_names:
    lora_module_names.remove("lm_head")
```

`lm_head` 是最终词表输出层，通常不作为 LoRA 注入目标。

## 9. 创建 LoRA 配置

`create_peft_config(modules)` 创建 LoRA 配置：

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
```

参数含义：

- `r=16`：LoRA 低秩矩阵的秩。秩越大，可训练容量越强，但参数量也越多。
- `lora_alpha=32`：LoRA 缩放系数，用于控制 adapter 输出强度。
- `target_modules=modules`：在哪些线性层上插入 LoRA adapter。
- `lora_dropout=0.1`：LoRA 分支 dropout，用于降低过拟合。
- `bias="none"`：不训练 bias 参数。
- `task_type="CAUSAL_LM"`：任务类型是因果语言建模。

接着把基础模型包装为 PEFT 模型：

```python
model = get_peft_model(model, peft_config)
```

此时基础模型主体参数保持冻结，训练过程中主要更新 LoRA adapter 参数。

脚本会打印可训练参数比例：

```python
print_trainable_parameters(model)
```

这一步用于确认训练的是少量 adapter 参数，而不是全量模型参数。

## 10. 使用 SFTTrainer 进行监督微调

训练器使用 TRL 的 `SFTTrainer`：

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(...),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
```

核心设置：

- `dataset_text_field="text"`：训练器读取预处理后生成的 `text` 字段。
- `max_seq_length=1024`：训练时最大序列长度。
- `DataCollatorForLanguageModeling(..., mlm=False)`：使用 causal LM 训练方式，而不是 masked LM。

训练目标可以理解为：

```text
给定 system + user + assistant 的完整文本，让模型学习预测下一个 token。
```

虽然训练 loss 覆盖整段文本，但实际希望模型学到的是：在推理时看到 system 和 user 后，能够生成类似 assistant 的内容，也就是自动驾驶轨迹规划答案。

## 11. 训练超参数

`TrainingArguments` 中的主要超参数如下：

```python
per_device_train_batch_size=1
gradient_accumulation_steps=1
weight_decay=0.01
warmup_ratio=0.1
learning_rate=2e-4
fp16=True
logging_steps=1
save_steps=10000
optim="adamw_torch"
lr_scheduler_type="linear"
num_train_epochs=15
```

含义：

- `per_device_train_batch_size=1`：每张 GPU 上每步处理 1 条样本。
- `gradient_accumulation_steps=1`：不额外累积梯度。
- `weight_decay=0.01`：使用权重衰减抑制过拟合。
- `warmup_ratio=0.1`：前 10% 训练步数进行学习率 warmup。
- `learning_rate=2e-4`：LoRA 微调学习率，通常高于全量微调学习率。
- `fp16=True`：使用半精度训练，降低显存占用。
- `logging_steps=1`：每一步打印训练日志。
- `save_steps=10000`：每 10000 步保存一次 checkpoint。
- `optim="adamw_torch"`：使用 PyTorch AdamW 优化器。
- `lr_scheduler_type="linear"`：使用线性学习率调度。
- `num_train_epochs=15`：训练集整体遍历 15 轮。

## 12. 关闭 cache 并检查参数类型

训练前脚本设置：

```python
model.config.use_cache = False
```

训练时关闭 cache 可以避免与梯度检查点冲突。推理阶段可以重新开启 cache 来加速生成。

随后脚本统计并打印模型参数 dtype 分布：

```python
for _, p in model.named_parameters():
    dtype = p.dtype
    ...
```

这一步用于检查量化加载、半精度和 LoRA 包装是否符合预期。

## 13. 启动训练

实际训练由以下代码启动：

```python
train_result = trainer.train()
```

训练完成后，脚本会记录指标和状态：

```python
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
```

这些信息会保存到 `output_dir`，便于后续查看训练 loss、训练步数和 trainer 状态。

## 14. 保存 LoRA adapter

训练结束后，脚本保存模型：

```python
os.makedirs(output_dir, exist_ok=True)
trainer.model.save_pretrained(output_dir)
```

需要注意的是，这里保存的是 PEFT/LoRA adapter 权重，而不是完整基础模型权重。推理时需要同时提供：

- 原始基础模型：`--model_name`
- 训练得到的 adapter 路径：`--adapter_path`

也就是说，训练输出目录通常会被 `inference.py` 作为 adapter checkpoint 加载。

## 15. 释放显存

最后脚本删除模型和 trainer，并清理 CUDA 缓存：

```python
del model
del trainer
torch.cuda.empty_cache()
```

这可以释放显存，方便继续运行推理、评估或其他训练任务。

## 16. 总体流程总结

`training.py` 的完整微调流程如下：

1. 从 `data/train_with_token.json` 读取 Chat 格式训练样本。
2. 把 `system/user/assistant` 三段消息拼成单个 `text` 字段。
3. 使用 tokenizer 将文本转换为模型输入 token。
4. 过滤过长样本并打乱训练集。
5. 使用 4-bit 量化加载基础 causal language model。
6. 准备 k-bit 量化模型训练。
7. 自动扫描 4-bit 线性层并插入 LoRA adapter。
8. 使用 `SFTTrainer` 进行监督微调。
9. 保存训练指标、状态和 LoRA adapter。
10. 推理时将基础模型和 adapter 一起加载，用于生成轨迹预测结果。
