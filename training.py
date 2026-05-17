import os
import argparse
from functools import partial
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import load_dataset, Dataset


def load_model(model_name, bnb_config, cache_dir, security_token):
    # 根据当前机器可见的 GPU 数量，让 transformers 自动把模型切分到可用设备上。
    # 这里配合 BitsAndBytesConfig 以 4-bit 量化方式加载基础大模型，
    # 可以显著降低显存占用，使 7B 级别模型更容易在单机 GPU 上微调。
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        max_memory={i: max_memory for i in range(n_gpus)},
        cache_dir=cache_dir,
        token=security_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, token=security_token
    )
    # LLaMA 等 causal LM tokenizer 默认可能没有 pad_token。
    # 训练 batch 需要 padding，因此直接复用 eos_token 作为 pad_token。
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompt_formats(sample, for_validation: bool = False):
    """把 create_data_split.py 生成的 messages 样本拼成 SFTTrainer 使用的 text 字段。

    原始样本包含 system、user、assistant 三条 Chat 消息。
    训练时需要把三条消息按固定顺序拼成一段纯文本，让 causal LM 学习
    在 system + user 条件下继续生成 assistant 答案。
    """
    message = sample["messages"]
    system = f"{message[0]['role']}\n{message[0]['content']}"
    user = f"{message[1]['role']}\n{message[1]['content']}"
    assistant = f"{message[2]['role']}\n{message[2]['content']}"
    end = "### End"

    if for_validation:
        # 验证/推理格式只保留 system 和 user，不拼入 assistant 标签。
        parts = [part for part in [system, user, end] if part]
    else:
        # 训练格式包含 assistant 标准答案，即 Thoughts、Meta Action 和 Trajectory。
        parts = [part for part in [system, user, assistant, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    # 不同模型配置中记录最大上下文长度的字段名可能不同。
    # 这里依次检查常见字段，找不到时默认使用 1024。
    conf = model.config
    max_length = None
    for length_setting in [
        "n_positions",
        "max_position_embeddings",
        "seq_length",
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """对一批 text prompt 做 tokenizer 编码。"""
    # 超过模型上下文长度的文本会被截断，避免训练时序列过长导致显存溢出。
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
    dataset: Dataset,
    for_validation: bool = False,
):
    """将原始 JSON/ndjson 数据集转换为可训练的 tokenized dataset。"""

    # 第一步：把每条样本的 messages 转成一个完整的 text prompt。
    print("Preprocessing dataset...")
    dataset = dataset.map(
        create_prompt_formats, fn_kwargs={"for_validation": for_validation}
    )

    # 第二步：批量 tokenize，把 text 转换为 input_ids、attention_mask 等模型输入。
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # 第三步：过滤掉长度仍然达到或超过 max_length 的样本，
    # 保留完整落在模型上下文窗口内的数据。
    dataset = dataset.filter(
        lambda sample: len(sample["input_ids"]) < max_length
    )

    # 第四步：打乱样本顺序，减少训练时数据顺序带来的偏差。
    dataset = dataset.shuffle(seed=seed)

    return dataset


def create_bnb_config():
    # QLoRA 风格的 4-bit 量化配置：
    # - load_in_4bit=True：以 4-bit 加载基础模型权重；
    # - nf4：适合正态分布权重的 4-bit 量化类型；
    # - double quant：进一步压缩量化常数；
    # - bfloat16：用于量化计算的数据类型。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules):
    """创建 LoRA 参数高效微调配置。"""
    config = LoraConfig(
        r=16,  # LoRA 低秩矩阵的秩，数值越大可训练容量越强。
        lora_alpha=32,  # LoRA 缩放系数，用于控制适配器更新强度。
        target_modules=modules,  # 只在这些线性层上插入 LoRA adapter。
        lora_dropout=0.1,  # LoRA 分支 dropout，降低过拟合风险。
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    # 基础模型以 4-bit 量化方式加载后，线性层类型为 bnb.nn.Linear4bit。
    # 这里自动扫描所有 4-bit 线性层名称，作为 LoRA 的 target_modules。
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head 是最终词表输出层，通常不挂 LoRA，避免影响输出头权重。
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir):
    # 开启梯度检查点，用更多计算换取更低显存占用。
    # 这对 7B 级别模型的单机微调很重要。
    model.gradient_checkpointing_enable()

    # 让量化模型进入可训练状态，例如处理梯度、输入嵌入和 layer norm 等细节。
    model = prepare_model_for_kbit_training(model)

    # 自动找到模型中所有 4-bit 线性层，后续只训练这些层上的 LoRA adapter。
    modules = find_all_linear_names(model)

    # 根据扫描到的线性层创建 LoRA 配置，并把基础模型包装成 PEFT 模型。
    # 基础模型主体权重保持冻结，只更新新增的 LoRA 参数。
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # 打印可训练参数比例，确认当前训练的是少量 LoRA 参数而不是全量模型。
    print_trainable_parameters(model)

    # SFTTrainer 会读取 dataset_text_field="text" 中的完整 prompt，
    # 使用 causal language modeling 目标训练模型续写 assistant 部分。
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        # Training parameters from here:
        # https://huggingface.co/blog/Llama2-for-non-engineers
        args=TrainingArguments(
            per_device_train_batch_size=1,  # 每张 GPU 上的 batch size。
            gradient_accumulation_steps=1,  # 梯度累积步数，用于扩大等效 batch size。
            weight_decay=0.01,  # 权重衰减，抑制过拟合。
            warmup_ratio=0.1,  # 前 10% 训练步数用于学习率 warmup。
            learning_rate=2e-4,  # LoRA 微调常用学习率，通常高于全量微调。
            fp16=True,  # 使用半精度训练以降低显存占用。
            logging_steps=1,  # 每步打印训练日志。
            save_steps=10000,  # 每 10000 步保存一次 checkpoint。
            output_dir=output_dir,
            optim="adamw_torch",  # "paged_adamw_32bit",
            lr_scheduler_type="linear",
            num_train_epochs=15,  # 遍历训练集 15 轮。
        ),
        # causal LM 训练不是 masked language modeling，因此 mlm=False。
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 训练时关闭 cache，避免和梯度检查点冲突；推理时可以重新开启以加速生成。
    model.config.use_cache = False

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # 训练前打印各数据类型参数占比，用于确认量化和 LoRA 包装是否符合预期。
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # 启动监督微调。训练目标是让模型在给定 system + user 的情况下，
    # 生成 assistant 中的 Thoughts、Meta Action 和未来轨迹点。
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # 保存最终 LoRA adapter。这里保存的是 PEFT adapter 权重，
    # 推理时需要和原始基础模型一起加载。
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # 释放显存，方便后续合并权重或运行其他脚本。
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file",
        "-t",
        dest="train_data_file",
        type=str,
        help="Training JSON data file.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        "-m",
        dest="model_name",
        type=str,
        help="Name of base model to train an adapter for.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        type=str,
        help="Path to save the trained adapter model.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        default=None,
        required=False,
        help="The cache directory to save the downloaded models.",
    )

    args = parser.parse_args()
    model_name = args.model_name
    training_data_file = args.train_data_file
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    hf_token = os.getenv("HF_ACCESS_TOKEN")

    # 读取 create_data_split.py 生成的训练文件。
    # 文件中每条样本都包含 messages 和 token，load_dataset 会把它们加载为 Dataset。
    dataset = load_dataset(
        "json",
        data_files=training_data_file,
        split="train",
    )
    print(f"Number of prompts: {len(dataset)}")
    print(f"Column names are: {dataset.column_names}")

    bnb_config = create_bnb_config()

    # 以 4-bit 量化方式加载基础 causal LM 和 tokenizer。
    model, tokenizer = load_model(model_name, bnb_config, cache_dir, hf_token)

    # 获取模型上下文长度，用于后续 tokenizer 截断和样本过滤。
    max_length = get_max_length(model)

    # 将 messages 格式训练数据转换为 text prompt，再 tokenize 成模型输入。
    dataset = preprocess_dataset(tokenizer, max_length, 0, dataset)

    # 执行 QLoRA/LoRA 监督微调，并把 adapter 保存到 output_dir。
    train(model, tokenizer, dataset, output_dir)
