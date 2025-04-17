# src/embedding/llm_embedding.py
"""为用户和职位生成LLM嵌入的模块，使用微调模型。"""

import logging
import torch
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from safetensors.torch import load_file
import warnings
import argparse

# 抑制特定警告
warnings.filterwarnings("ignore", message="找到缺失的适配器键")

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, lora_path: str, device: str = "cuda") -> tuple:
    """加载基础模型，应用LoRA适配器，并合并权重。"""
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        lora_model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

        state_dict = load_file(f"{lora_path}/adapter_model.safetensors")
        adapted_state_dict = {
            k.replace("base_model.model.model", "base_model.model"): v for k, v in state_dict.items()
        }
        lora_model.load_state_dict(adapted_state_dict, strict=False)

        merged_model = lora_model.merge_and_unload()
        merged_model.config.pad_token_id = 151643
        merged_model.to(device).eval()

        return merged_model, tokenizer
    except Exception as e:
        logger.error(f"加载模型或分词器失败: {e}")
        raise


def get_eos_embedding(text: str, tokenizer: AutoTokenizer, model: AutoModel, device: str = "cuda") -> torch.Tensor:
    """提取输入文本的EOS token嵌入或平均嵌入。"""
    inputs = tokenizer(
        text + tokenizer.eos_token,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=384
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]
    eos_token_id = tokenizer.eos_token_id
    eos_token_index = (inputs["input_ids"] == eos_token_id) & (inputs["attention_mask"] == 1)
    eos_token_index = eos_token_index.nonzero(as_tuple=True)

    if eos_token_index[0].numel() > 0:
        return last_hidden_state[eos_token_index]
    else:
        valid_hidden_states = last_hidden_state * inputs["attention_mask"].unsqueeze(-1)
        seq_length = inputs["attention_mask"].sum(dim=1, keepdim=True)
        return valid_hidden_states.sum(dim=1) / seq_length


def generate_embeddings(
        dataset_path: str,
        model_path: str,
        lora_path: str,
        user_output_path: str,
        job_output_path: str,
        device: str = "cuda"
) -> None:
    """为用户和职位生成并保存LLM嵌入。"""
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, lora_path, device)
        dataset = load_from_disk(dataset_path)

        # 处理唯一用户
        unique_users = (
            dataset.filter(lambda x: x["user"] is not None and x["Summary"] is not None)
            .select_columns(["user", "Summary"])
            .to_pandas()
            .drop_duplicates()
            .sort_values(by="user")
        )
        unique_users_dataset = Dataset.from_pandas(unique_users)

        logger.info("正在处理用户嵌入...")
        user_embeddings = {
            user: get_eos_embedding(resume, tokenizer, model, device)
            for user, resume in tqdm(
                zip(unique_users_dataset["user"], unique_users_dataset["Summary"]),
                total=len(unique_users_dataset),
                desc="用户"
            )
        }
        torch.save(user_embeddings, user_output_path)
        logger.info(f"用户嵌入已保存到 {user_output_path}")

        # 处理唯一职位
        unique_jobs = (
            dataset.filter(lambda x: x["job_id"] is not None and x["job_summary"] is not None)
            .select_columns(["job_id", "job_summary"])
            .to_pandas()
            .drop_duplicates()
            .sort_values(by="job_id")
        )
        unique_jobs_dataset = Dataset.from_pandas(unique_jobs)

        logger.info("正在处理职位嵌入...")
        job_embeddings = {
            job: get_eos_embedding(job_desc, tokenizer, model, device)
            for job, job_desc in tqdm(
                zip(unique_jobs_dataset["job_id"], unique_jobs_dataset["job_summary"]),
                total=len(unique_jobs_dataset),
                desc="职位"
            )
        }
        torch.save(job_embeddings, job_output_path)
        logger.info(f"职位嵌入已保存到 {job_output_path}")
    except Exception as e:
        logger.error(f"生成嵌入时出错: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成LLM嵌入")
    parser.add_argument("--dataset_path", type=str, required=True, help="处理后的数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA权重路径")
    parser.add_argument("--user_output_path", type=str, required=True, help="保存用户嵌入的路径")
    parser.add_argument("--job_output_path", type=str, required=True, help="保存职位嵌入的路径")
    args = parser.parse_args()

    generate_embeddings(
        args.dataset_path, args.model_path, args.lora_path, args.user_output_path, args.job_output_path
    )