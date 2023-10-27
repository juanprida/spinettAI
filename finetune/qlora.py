"""Finetune Llama 2 7B model using Quantized Low Rank Adaptaion (QLORA) with HuggingFace."""

import json
import os

import datasets
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "NousResearch/Llama-2-7b-chat-hf-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)


class InstructLyricsDataset:
    """Dataset for instruct fine tuning."""

    def __init__(self, input_file_path: str) -> None:
        self.input_file_path = input_file_path
        self.data = self.load_data()

    def load_data(self):
        """Load data from file."""
        with open(self.input_file_path, "r") as f:
            data = json.load(f)

        return [{"text": lyrics} for _, lyrics in data.items()]

    def _transform_text_for_finetuning(self, text: str) -> str:
        """Transform text for finetuning."""
        text_0, text_1 = text.split("\n", maxsplit=1)
        instruction = f"<s>[INST] Escribe una cancion siguiendo la siguiente frase: \n{text_0}[/INST]"
        answer = f"{text_1} </s>"
        return instruction + answer

    def get(self) -> datasets.Dataset:
        """Get dataset for instruct fine tuning."""
        pd_data = pd.DataFrame(self.data)
        dataset = datasets.Dataset.from_pandas(pd_data)
        dataset = dataset.map(
            lambda x: {"text": self._transform_text_for_finetuning(x["text"])}
        )
        return dataset


if __name__ == "__main__":
    dir_location = os.path.split(os.path.dirname(os.path.abspath("__file__")))[0]
    input_file_path = os.path.join(dir_location, "data/lyrics.json")
    dataset = InstructLyricsDataset(input_file_path=input_file_path).get()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)
    trainer.model.push_to_hub(new_model)
