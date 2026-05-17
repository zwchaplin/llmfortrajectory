import argparse
import pickle
import ndjson
import json
import tiktoken
from prompt_message import (
    system_message,
    generate_user_message,
    generate_assistant_message,
)

def save_data_split(data: list, tokens: list, encoding_model: str, save_file: str) -> None:
    """根据 token 列表生成一份训练或验证数据，并保存为 ndjson 文件。

    Args:
        data: 从原始 pickle 文件中读取的完整轨迹数据。
        tokens: 当前数据 split 中需要导出的样本 token 列表。
        encoding_model: 用于统计 prompt token 数的模型名称。
            会传入 tiktoken.encoding_for_model(encoding_model)。
        save_file: 生成后的 ndjson 数据保存路径。
    """
    print(f"Saving data split: {len(tokens)} : {save_file}")

    encoding = tiktoken.encoding_for_model(encoding_model)

    num_language_tokens = 0
    num_system_tokens = 0
    num_user_tokens = 0
    num_assistant_tokens = 0

    traj_only = False

    train_messages = []
    num_samples = len(tokens)
    for token_i, token in enumerate(tokens):
        if token_i >= num_samples:
            break
        # 每个 token 对应原始数据中的一个场景/样本。
        # generate_user_message 会读取该样本的历史轨迹、地图或上下文信息，
        # 将其组织成发给模型的用户问题。
        user_message = generate_user_message(data, token)

        # generate_assistant_message 会读取同一个样本的未来轨迹标签，
        # 生成模型需要学习输出的 assistant 回复，即监督微调目标。
        assitant_message = generate_assistant_message(data, token, traj_only=traj_only)

        # 统计 system/user/assistant 三部分文本的 token 数，
        # 用于估算微调数据规模和训练成本，不影响最终样本内容。
        num_language_tokens += len(encoding.encode(system_message))
        num_system_tokens += len(encoding.encode(system_message))
        num_language_tokens += len(encoding.encode(user_message))
        num_user_tokens += len(encoding.encode(user_message))
        num_language_tokens += len(encoding.encode(assitant_message))
        num_assistant_tokens += len(encoding.encode(assitant_message))

        # OpenAI Chat 微调格式：每行是一个 JSON 对象，
        # messages 中依次包含固定系统提示、由原始样本生成的用户输入、
        # 以及由未来轨迹标签生成的标准答案。
        train_message = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assitant_message},
            ],
            "token": token,
        }
        train_messages.append(train_message)

    print("#### Cost Summarization ####")
    print(f"Number of system tokens: {num_system_tokens}")
    print(f"Number of user tokens: {num_user_tokens}")
    print(f"Number of assistant tokens: {num_assistant_tokens}")
    print(f"Number of total tokens: {num_language_tokens}")

    # ndjson.dump 会把 train_messages 写成逐行 JSON，
    # 下游 training.py 可直接按行读取并用于 LoRA 微调。
    with open(save_file, "w", encoding="utf-8") as f:
        ndjson.dump(train_messages, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./create_data_split.py")
    parser.add_argument(
        '--model_name', '-m',
        dest="model_name",
        type=str,
        help = "Model name, used for encoding the messages.",
        required=True
    )
    parser.add_argument(
        '--data_file', '-d',
        dest="data_file",
        type=str,
        help="JSON data file. Will be split to train and validation.",
        required=True
    )
    parser.add_argument(
        '--split_data_file', '-s',
        dest="split_data_file",
        type=str,
        help="File containing data split.",
        required=True
    )
    parser.add_argument(
        '--val_data_file', '-v',
        dest="val_data_file",
        type=str,
        help="Path to save validation JSON data file",
        required=True
    )
    parser.add_argument(
        '--train_data_file', '-t',
        dest="train_data_file",
        type=str,
        help="Path to save validation JSON data file",
        required=True
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    model_name = FLAGS.model_name
    data_file = FLAGS.data_file
    split_data_file = FLAGS.split_data_file
    val_data_file = FLAGS.val_data_file
    train_data_file = FLAGS.train_data_file

    # 读取 GPT-Driver 风格的原始数据和预先生成的数据划分。
    # data_file 是 pickle 格式，保存完整样本内容；
    # split_data_file 是 JSON 格式，只保存 train/val 对应的样本 token。
    data = pickle.load(open(data_file, "rb"))
    split = json.load(open(split_data_file, "r", encoding="utf-8"))

    # 根据 split 文件取得训练集和验证集 token。
    # 脚本不会重新随机划分数据，因此只要 split 文件不变，
    # 生成的 train/val 数据就是可复现的。
    train_tokens = split["train"]
    val_tokens = split["val"]

    # 分别将训练集和验证集 token 转换成 chat 微调样本，
    # 输出文件通常对应 data/train_with_token.json 和 data/val_with_token.json。
    save_data_split(data, train_tokens, model_name, train_data_file)
    save_data_split(data, val_tokens, model_name, val_data_file)
