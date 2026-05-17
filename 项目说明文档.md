# 项目说明文档

## 训练集和验证集文档如何生成

本项目使用 GPT-Driver 风格的原始轨迹数据，并通过 `create_data_split.py` 生成模型训练和验证所需的 Chat 微调数据。仓库中已经提供了生成后的文件：

- `data/train_with_token.json`：训练集数据。
- `data/val_with_token.json`：验证集数据。

如果需要从原始数据重新生成这两个文件，需要准备两个输入：

- `--data_file`：原始数据文件，pickle 格式，保存完整轨迹样本内容。
- `--split_data_file`：数据划分文件，JSON 格式，保存训练集和验证集对应的样本 `token`。

生成过程如下：

1. `create_data_split.py` 先读取 pickle 原始数据和 JSON 划分文件。
2. 从划分文件中读取 `split["train"]` 和 `split["val"]`，分别得到训练集 token 列表和验证集 token 列表。
3. 脚本不会重新随机划分数据，而是严格按照 split 文件中的 token 生成数据，因此只要 split 文件不变，生成结果就是可复现的。
4. 对每个 token，脚本会在原始数据中找到对应场景样本，并调用 `prompt_message.py` 中的函数生成三段 Chat 消息：
   - `system_message`：固定系统提示，定义任务背景和模型角色。
   - `generate_user_message(data, token)`：根据该样本的历史轨迹、地图或上下文信息生成用户输入。
   - `generate_assistant_message(data, token)`：根据该样本的未来轨迹标签生成标准答案，也就是模型需要学习输出的内容。
5. 每个样本最终被保存为如下结构：

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

6. 脚本使用 `tiktoken` 统计 system、user、assistant 三部分文本的 token 数，用于估算数据规模和训练成本。
7. 最后，训练集和验证集分别以 ndjson 形式写入输出文件。ndjson 表示每一行都是一个独立 JSON 样本，方便后续训练脚本逐行读取。

## `prompt_message.py` 如何构造提示词

`prompt_message.py` 是数据转换的核心文件。它不直接读取磁盘文件，而是接收 `create_data_split.py` 已经加载好的原始数据对象 `data` 和样本标识 `token`。每次处理一个样本时，脚本会先通过 `data[token]` 取出该场景对应的原始字段，然后分别构造 system、user 和 assistant 三类消息。

### 1. System Message：定义任务和输出格式

文件开头定义了固定的 `system_message`，用于告诉模型当前任务是自动驾驶轨迹规划。它包含以下信息：

- 模型角色：模型被设定为自动驾驶车辆的大脑。
- 坐标系：自车位于 `(0, 0)`，X 轴表示横向方向，Y 轴表示车辆前进方向。
- 规划目标：生成未来 3 秒的轨迹，共 6 个 waypoint，每 0.5 秒一个点。
- 输入内容：感知与预测、历史轨迹、ego 状态、任务目标。
- 输出格式：要求模型输出思考过程、Meta Action 和最终 Trajectory。

在最终训练样本中，`system_message` 会作为第一条消息：

```json
{"role": "system", "content": system_message}
```

### 2. User Message：把原始场景数据转换成模型输入

`generate_user_message(data, token, perception_range=20.0, short=True)` 用于构造用户输入。它的第一步是取出当前样本：

```python
data_dict = data[token]
```

之后从 `data_dict` 中提取原始字段，并按固定文本模板写入 `user_message`。

#### 2.1 感知和预测信息

感知部分主要使用以下字段：

- `gt_boxes`：周围目标当前检测框，形状通常为 `[N, 7]`，其中前两维是目标相对自车的位置 `(x, y)`。
- `gt_names`：周围目标类别名称。
- `gt_agent_fut_trajs`：周围目标未来轨迹的相对位移，代码中 reshape 为 `[N, 6, 2]`。
- `gt_agent_fut_masks`：周围目标未来轨迹的有效性 mask，表示某个时间步的预测位置是否有效。

原始的 `gt_agent_fut_trajs` 保存的是相对位移，不是绝对坐标。代码会先做累加，再加上目标当前坐标，得到目标未来 6 个时间步的绝对位置：

```python
object_rel_fut_trajs = data_dict["gt_agent_fut_trajs"].reshape(-1, 6, 2)
object_fut_trajs = np.cumsum(object_rel_fut_trajs, axis=1) + object_boxes[:, None, :2]
```

然后脚本会过滤一部分目标，避免提示词过长或包含无关对象：

- 如果目标当前和未来始终在自车后方，即 Y 坐标都小于等于 0，则跳过。
- 如果目标当前或未来位置超出 `perception_range`，默认 20 米，则跳过。

在默认 `short=True` 的情况下，每个保留目标会被写成简短描述：

```text
Perception and Prediction:
 - car at (x,y), moving to (x,y).
```

如果该目标最后一个预测时间步无效，则写为：

```text
moving to unknown location.
```

因此，感知部分的作用是把结构化目标检测和目标预测结果压缩成自然语言，让模型知道周围有哪些对象、它们在哪里、未来大致移动到哪里。

#### 2.2 Ego 状态信息

Ego 状态来自 `gt_ego_lcf_feat` 和 `gt_ego_his_diff`：

- `gt_ego_lcf_feat[0]`、`gt_ego_lcf_feat[1]`：自车速度分量，代码中乘以 `0.5`，转换成每 0.5 秒位移尺度。
- `gt_ego_lcf_feat[4]`：航向角速度 `v_yaw`。
- `gt_ego_lcf_feat[2]`、`gt_ego_lcf_feat[3]`：can bus 相关状态。
- `gt_ego_lcf_feat[7]`：heading speed，代码中同样乘以 `0.5`。
- `gt_ego_lcf_feat[8]`：方向盘或转向信号。
- `gt_ego_his_diff`：历史轨迹差分，用最后两个差分点估计当前加速度。

加速度计算方式为：

```python
ax = data_dict["gt_ego_his_diff"][-1, 0] - data_dict["gt_ego_his_diff"][-2, 0]
ay = data_dict["gt_ego_his_diff"][-1, 1] - data_dict["gt_ego_his_diff"][-2, 1]
```

最终写入 prompt 的格式类似：

```text
Ego-States:
 - Velocity (vx,vy): (...)
 - Heading Angular Velocity (v_yaw): (...)
 - Acceleration (ax,ay): (...)
 - Can Bus: (...)
 - Heading Speed: (...)
 - Steering: (...)
```

这部分让模型知道自车当前运动状态，例如速度、加速度、转向趋势和车身状态。

#### 2.3 历史轨迹信息

历史轨迹来自：

- `gt_ego_his_trajs`：自车过去 2 秒轨迹。
- `gt_ego_his_diff`：自车过去 2 秒轨迹差分。

`generate_user_message` 主要读取 `gt_ego_his_trajs` 的前 4 个点，并写成：

```text
Historical Trajectory (last 2 seconds): [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
```

这部分提供自车过去运动趋势，使模型能够根据历史轨迹判断车辆是在直行、变道、转弯、减速还是停止。

#### 2.4 Mission Goal：未来驾驶目标

任务目标来自：

- `gt_ego_fut_cmd`

这是一个三维命令向量，代码按如下规则转换为文本：

- 第一个维度大于 0：`RIGHT`
- 第二个维度大于 0：`LEFT`
- 否则第三个维度应大于 0：`FORWARD`

写入 prompt 时每个样本只会出现其中一个目标，例如：

```text
Mission Goal: RIGHT
```

或：

```text
Mission Goal: LEFT
```

这部分告诉模型未来 3 秒的高层驾驶意图。

### 3. Assistant Message：把未来真实轨迹转换成监督答案

`generate_assistant_message(data, token, traj_only=False)` 用于生成训练时的标准答案。它同样先取出当前样本：

```python
data_dict = data[token]
```

如果 `traj_only=False`，脚本会先调用 `generate_chain_of_thoughts(data_dict)` 生成简化的思考过程和 Meta Action，然后再附加真实未来轨迹。`create_data_split.py` 中目前设置 `traj_only = False`，因此训练数据默认包含 Thoughts、Meta Action 和 Trajectory。

#### 3.1 生成 Thoughts：基于规则识别潜在碰撞目标

`generate_chain_of_thoughts` 会用简单规则构造类似推理过程的文本。它首先读取自车历史和未来轨迹：

- `gt_ego_fut_trajs`：自车未来轨迹。
- `gt_ego_his_trajs`：自车历史轨迹。
- `gt_ego_fut_diff`：自车未来轨迹差分。
- `gt_ego_his_diff`：自车历史轨迹差分。

然后根据当前速度和加速度估计自车未来位置：

```python
ego_estimate_velos = [
    [0, 0],
    [vx, vy],
    [vx+ax, vy+ay],
    ...
]
ego_estimate_trajs = np.cumsum(ego_estimate_velos, axis=0)
```

同时，它会把周围目标的未来相对位移转换为绝对轨迹，并把当前目标位置拼接到未来轨迹前面，使目标轨迹包含当前时刻和未来 6 个时间步，共 7 个点。

接着，脚本对每个目标、每个时间步做规则碰撞检测：

```python
collision_detection(
    ego_x, ego_y, 0.925, 2.04,
    object_x, object_y, size_x, size_y
)
```

其中 `0.925` 和 `2.04` 可以理解为自车半宽、半长；目标尺寸来自 `gt_boxes[i, 3:5] * 0.5`。碰撞判断还额外加入安全距离：

- 横向安全距离 `x_space=1.0`
- 纵向安全距离 `y_space=3.0`

如果没有发现潜在碰撞目标，assistant 中会写：

```text
Thoughts:
 - Notable Objects from Perception: None
   Potential Effects from Prediction: None
```

如果发现某个目标在未来时间步进入自车安全区域，则写入该目标的位置和影响：

```text
Thoughts:
 - Notable Objects from Perception: car at (x,y)
   Potential Effects from Prediction: within the safe zone of the ego-vehicle at the 1.5-second timestep
```

这部分不是由模型推理得到的，而是由原始标签和规则自动生成的监督文本。

#### 3.2 生成 Meta Action：根据未来轨迹归纳驾驶动作

`generate_meta_action` 根据自车未来轨迹和速度变化生成高层动作描述。它分两步判断：

第一步判断速度行为：

- 当前和未来末速度都很小：`STOP`
- 未来末速度接近 0：减速到停止
- 未来末速度与当前速度差异很小：匀速
- 未来末速度更小：减速或快速减速
- 未来末速度更大：加速或快速加速

第二步判断横向行为：

- 如果未来轨迹所有 x 坐标都接近 0：直行。
- 如果最终 x 坐标小于 0：向左变道或左转。
- 如果最终 x 坐标大于 0：向右变道或右转。
- 横向位移绝对值超过 `4.0` 时认为是转弯，否则认为是变道。

最终 Meta Action 会被转成大写，例如：

```text
Meta Action: MOVE FORWARD WITH A CONSTANT SPEED
```

或：

```text
Meta Action: CHANGE LANE TO RIGHT WITH AN ACCELERATION
```

#### 3.3 生成 Trajectory：使用未来真实轨迹作为标签

最终轨迹标签来自：

- `gt_ego_fut_trajs`

代码读取索引 1 到 6 的 6 个未来轨迹点：

```python
x1 = data_dict["gt_ego_fut_trajs"][1][0]
...
x6 = data_dict["gt_ego_fut_trajs"][6][0]
```

然后写成模型需要输出的格式：

```text
Trajectory:
[(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]
```

这 6 个点对应未来 3 秒轨迹，每 0.5 秒一个 waypoint，是训练中最重要的监督目标。

### 4. 最终样本的提示词结构

经过 `prompt_message.py` 处理后，一个训练样本实际由三部分组成：

```text
system:
  固定自动驾驶轨迹规划任务说明。

user:
  Perception and Prediction
  Ego-States
  Historical Trajectory
  Mission Goal

assistant:
  Thoughts
  Meta Action
  Trajectory
```

也就是说，原始结构化数据没有直接喂给模型，而是先被转换成自然语言描述。训练时，模型学习从 `system + user` 中理解场景，并生成 `assistant` 中的思考摘要、驾驶动作和未来 6 个轨迹点。

示例命令：

```bash
python create_data_split.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --data_file "<原始pickle数据路径>" \
  --split_data_file "<数据划分json路径>" \
  --train_data_file "data/train_with_token.json" \
  --val_data_file "data/val_with_token.json"
```

在 Windows PowerShell 中，可以写成一行命令：

```powershell
python create_data_split.py --model_name "meta-llama/Llama-2-7b-hf" --data_file "<原始pickle数据路径>" --split_data_file "<数据划分json路径>" --train_data_file "data/train_with_token.json" --val_data_file "data/val_with_token.json"
```

## 项目用途

本项目用于基于大语言模型进行轨迹预测实验。核心思路是把轨迹预测样本转换成 Chat 格式的文本数据，然后使用 LoRA 适配器对开源语言模型进行参数高效微调。训练后的模型可以根据历史轨迹和场景上下文预测未来轨迹，并通过评估脚本计算结果指标。

## 目录结构

- `create_data_split.py`：从 GPT-Driver 风格原始数据生成训练集和验证集。
- `prompt_message.py`：构造 system、user、assistant prompt。
- `training.py`：使用 LoRA 对基础模型进行微调。
- `inference.py`：加载基础模型和 LoRA checkpoint，在验证集上生成预测结果。
- `evaluation.py`：评估预测 JSON 文件。
- `data/`：保存训练集和验证集数据。
- `checkpoints/`：建议用于保存训练得到的 LoRA checkpoint。
- `output/`：建议用于保存推理输出。
- `results/`：建议用于保存评估结果。

## 环境配置

建议使用 Python 3.9：

```bash
conda create -n llmtp python=3.9
conda activate llmtp
pip install -r requirements.txt
```

部分 Hugging Face 模型可能需要申请访问权限或配置 token。不要把 token 写入源码文件，建议保存在 shell 环境变量或本地 Hugging Face 配置中。

## 训练

训练脚本会读取 `data/train_with_token.json`，加载指定 Hugging Face 基础模型，并将 LoRA adapter 保存到 `checkpoints/` 目录。

Linux/macOS 可运行：

```bash
./run_training.sh
```

该脚本实际执行：

```bash
python training.py \
  --train_data_file "data/train_with_token.json" \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_path "checkpoints/llama2_lora"
```

Windows PowerShell 可直接执行同样参数的一行命令。

## 推理

推理脚本会读取验证集，加载基础模型和 LoRA adapter checkpoint，并把预测结果写入 `output/`。

Linux/macOS 可运行：

```bash
./run_inference.sh
```

该脚本实际执行：

```bash
python inference.py \
  --validation_data_file "data/val_with_token.json" \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --adapter_path "checkpoints/llama2_lora/checkpoint-70000-llama2" \
  --results_file "output/llama2_lora/output_llama2_lora.json"
```

运行前需要确认 `--adapter_path` 指向实际存在的 checkpoint。

## 评估

评估脚本会读取预测 JSON 文件，并把评估结果保存到 `results/`。

Linux/macOS 可运行：

```bash
./run_evaluation.sh
```

也可以单独评估某个预测文件：

```bash
python evaluation.py \
  --prediction_file "output/llama2_lora/output_llama2_lora.json" \
  --output_path "results/llama2_lora/output_llama2_lora.eval.json"
```

## 推荐工作流

1. 准备环境并安装依赖。
2. 如已有 `data/train_with_token.json` 和 `data/val_with_token.json`，可直接训练；否则先运行 `create_data_split.py` 生成数据。
3. 运行 `training.py` 训练 LoRA adapter。
4. 运行 `inference.py` 在验证集上生成预测结果。
5. 运行 `evaluation.py` 评估预测结果。

## 注意事项

- 不要提交大型模型 checkpoint、推理输出或评估结果，除非它们很小且可复现。
- `checkpoints/`、`output/`、`results/` 可按需在本地创建。
- 如果更换基础模型，需要同步调整训练、推理和 token 统计中使用的 `--model_name`。
