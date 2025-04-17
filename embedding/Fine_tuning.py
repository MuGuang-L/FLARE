# src/embedding/fine_tuning.py
"""使用LoRA微调语言模型以生成嵌入的模块。"""

import logging
import os
import torch
import numpy as np
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
from unsloth import FastLanguageModel
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DualModel(torch.nn.Module):
    """用于对比学习的自定义模型，带有LoRA微调。"""

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name_or_path,
            max_seq_length=384,
            dtype=torch.float16,
            load_in_4bit=True,
            output_hidden_states=True
        )
        self.model.config.pad_token_id = 151643
        self.cosine_embedding_loss = CosineEmbeddingLoss(margin=0.2)
        self.cos_weight = 0.7
        self.temperature = 0.1
        self.sigmoid = torch.nn.Sigmoid()

        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.peft = get_peft_model(self.model, config)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.size(0)
        num_sequences = input_ids.size(1)
        all_outputs = [
            self.peft(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                output_hidden_states=True,
                **kwargs
            ).hidden_states[-1]
            for i in range(num_sequences)
        ]

        eos_hidden_states = [
            self._get_eos_hidden(
                all_outputs[i], input_ids[:, i], self.model.config.eos_token_id, self.model.config.pad_token_id
            )
            for i in range(num_sequences)
        ]

        contrast_loss = (
                                self.info_nce_loss(eos_hidden_states[2], eos_hidden_states[0], eos_hidden_states[4],
                                                   0.1) +
                                self.info_nce_loss(eos_hidden_states[3], eos_hidden_states[1], eos_hidden_states[5],
                                                   0.1)
                        ) / 2
        similarity = torch.nn.functional.cosine_similarity(eos_hidden_states[0], eos_hidden_states[1], dim=-1)

        if labels is not None:
            match_loss = self.cosine_embedding_loss(eos_hidden_states[0], eos_hidden_states[1], labels)
            loss = match_loss * self.cos_weight + contrast_loss * (1 - self.cos_weight)
            return loss, similarity
        return similarity,

    def info_nce_loss(self, anchor, positive, negative, temperature):
        """计算InfoNCE损失。"""
        sim_ap = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        sim_an = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        logits = torch.clamp(
            torch.exp(sim_ap / temperature) / (
                    torch.exp(sim_ap / temperature) + torch.exp(sim_an / temperature)
            ),
            1e-7,
            1 - 1e-7
        )
        return -torch.log(logits + 1e-8).mean()

    def _get_eos_hidden(self, hidden_states, input_ids, eos_token_id, pad_token_id=None):
        """提取EOS token的隐藏状态或平均隐藏状态。"""
        eos_mask = (input_ids == eos_token_id)
        batch_size = hidden_states.size(0)
        eos_indices = eos_mask.long().argmax(dim=1)
        valid_eos_mask = eos_mask.any(dim=1)

        if pad_token_id is not None:
            valid_mask = (input_ids != pad_token_id)
            valid_hidden = hidden_states * valid_mask.unsqueeze(-1).float()
            avg_hidden = valid_hidden.sum(1) / valid_mask.sum(1).clamp(min=1).unsqueeze(-1)
        else:
            avg_hidden = hidden_states.mean(dim=1)

        return torch.where(
            valid_eos_mask.unsqueeze(-1),
            hidden_states[torch.arange(batch_size), eos_indices.clamp(max=hidden_states.size(1) - 1)],
            avg_hidden
        )


def process_function(examples: dict, tokenizer) -> dict:
    """处理数据集样本以供模型训练。"""
    sentences = []
    labels = []
    for sen1, sen2, sen3, sen4, sen5, sen6, label in zip(
            examples["Summary"],
            examples["job_summary"],
            examples["resume_section"],
            examples["job_desc"],
            examples["other_resume_summary"],
            examples["other_job_summary"],
            examples["label"]
    ):
        sentences.extend([
            sen1 + tokenizer.eos_token,
            sen2 + tokenizer.eos_token,
            sen3 + tokenizer.eos_token,
            sen4 + tokenizer.eos_token,
            sen5 + tokenizer.eos_token,
            sen6 + tokenizer.eos_token
        ])
        labels.append(1 if int(label) == 1 else -1)

    tokenized = tokenizer(
        sentences,
        max_length=384,
        truncation=True,
        padding="max_length"
    )
    tokenized = {k: [v[i:i + 6] for i in range(0, len(v), 6)] for k, v in tokenized.items()}
    tokenized["labels"] = labels
    return tokenized


def compute_metrics(eval_pred):
    """计算评估指标。"""
    predictions, labels = eval_pred
    probs = predictions
    y_true = np.array(labels)
    y_true = (y_true > 0).astype(int)

    best_threshold = 0.5
    best_f1 = 0
    for th in np.arange(0.1, 0.9, 0.05):
        preds = (probs > th).astype(int)
        current_f1 = f1_score(y_true, preds)
        if current_f1 > best_f1:
            best_threshold = th
            best_f1 = current_f1

    preds = (probs > best_threshold).astype(int)
    return {
        "准确率": accuracy_score(y_true, preds),
        "F1分数": best_f1,
        "精确率": precision_score(y_true, preds),
        "召回率": recall_score(y_true, preds),
        "最佳阈值": best_threshold
    }


def train_model(model_name_or_path: str, dataset_path: str, output_dir: str) -> None:
    """使用LoRA微调模型。"""
    try:
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name_or_path,
            max_seq_length=384,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        datasets = load_from_disk(dataset_path).filter(lambda x: x['label'] is not None)
        datasets = datasets.train_test_split(test_size=0.2, seed=2025)

        tokenized_datasets = datasets.map(
            lambda examples: process_function(examples, tokenizer),
            batched=True,
            remove_columns=datasets["train"].column_names
        )

        model = DualModel(model_name_or_path)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            max_grad_norm=1,
            save_total_limit=3,
            learning_rate=5e-6,
            weight_decay=0.01,
            metric_for_best_model="精确率",
            load_best_model_at_end=True,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            adam_epsilon=1e-6,
            optim="adamw_hf",
            fp16=True,
            fp16_opt_level="O1",
            fp16_full_eval=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.evaluate(tokenized_datasets["test"])
        logger.info(f"模型训练完成并保存到 {output_dir}")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA微调LLM")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, required=True, help="训练模型的输出目录")
    args = parser.parse_args()

    train_model(args.model_name_or_path, args.dataset_path, args.output_dir)