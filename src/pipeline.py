import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset,load_dataset, interleave_datasets, IterableDataset, concatenate_datasets
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
import os
from pathlib import Path
import swanlab
import logging
from .utils import generate_qa_prompt, process_func, process_qa_prompt
from tqdm import tqdm


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

class LengthIterableDataset(IterableDataset):
    def __init__(self, dataset, total_examples):
        self.dataset = dataset
        self.total_examples = total_examples

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.total_examples

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def cache_tokenized_data(dataset, tokenizer, config, cache_dir="tokenized_cache"):
    """
    Process and cache tokenized data one example at a time, saving to a single JSONL file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "tokenized_data.jsonl"

    total_examples = count_lines(f'{config["dataset_path"]}/train.jsonl')
    processed_count = 0

    with open(cache_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            processed = process_func(example, tokenizer, max_length=config["max_length"])
            
            json.dump(processed, f, ensure_ascii=False)
            f.write('\n')

            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{total_examples} examples")

    print(f"Saved {processed_count} examples to {cache_file}")

    return load_dataset("json", data_files=str(cache_file))["train"]

class Chatbot_Pipeline(object):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False, trust_remote_code=True)
        
        quantization_config = None
        if config.get("allow_quantization", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  
                bnb_4bit_compute_dtype=torch.float16,  
                bnb_4bit_use_double_quant=True,  
                bnb_4bit_quant_type="nf4"  
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2"
        )
        
        self.get_dataset(config["dataset"], config["training"], config["with_test"])
        self.max_new_tokens = config["max_new_tokens"]
        self.training = config["training"]

        if config["training"]:
            self.model.enable_input_require_grads()
        
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
                r=config["lora"]["r"],
                lora_alpha=config["lora"]["lora_alpha"],
                lora_dropout=config["lora"]["lora_dropout"]
            )
            self.model = get_peft_model(self.model, lora_config)
            
            config["TrainingArgs"]["output_dir"] = os.path.join(config["TrainingArgs"]["output_dir"], config["model_name"])
            args = TrainingArguments(**config["TrainingArgs"])
            
            swanlab_callback = SwanLabCallback(
                project=config["swanlab"]["project"],
                experiment_name=f"{config['model_name']}-QA",
                description=f"使用{config['model_name']}在问答数据集上微调。",
                config={
                    "model": config["model_name"],
                    "dataset": config["dataset"]["name"],
                }
            )
            
            
            self.trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=self.train_dataset,
                data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
                callbacks=[swanlab_callback],
            )
            
            
            self.with_test = config["with_test"]
        if config["resume_from_checkpoint"] is not None:
            if self.training:
                print(f"resuming frome {config['resume_from_checkpoint']}")
                self.trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])
            else:
                print(f"Loading checkpoint {config['resume_from_checkpoint']} for testing")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config["resume_from_checkpoint"],
                    is_trainable=False  
                )


    def get_dataset(self, config, training=True, with_test=False):


        if training:
            train_path = f'{config["dataset_path"]}/train.jsonl'
            train_ds = load_dataset("json", data_files=train_path, streaming=True)["train"]

            cache_dir = Path(config["dataset_path"])
            cache_file = cache_dir / "tokenized_data.jsonl"
            if cache_file.exists():
                print("Loading cached tokenized data...")
                self.train_dataset = load_dataset("json", data_files=str(cache_file))["train"]
            else:
                print("Caching tokenized data...")
                self.train_dataset = cache_tokenized_data(
                    dataset=train_ds,
                    tokenizer=self.tokenizer,
                    config=config,
                    cache_dir=cache_dir
                )

        if not training or with_test:
            self.test_dataset = pd.read_json(f'{config["dataset_path"]}/TruthReader_training_data_sample.json')

    def train(self):
        self.trainer.train()
        if self.with_test:
            self.test()
    
    def test(self):
        results = []
        for index, row in tqdm(self.test_dataset.iterrows(), total=len(self.test_dataset)):
            qa_instruction = process_qa_prompt(
                question=row["question"],
                documents=row["documents"],
                tokenizer=self.tokenizer,
                max_context_len=3000
            )
            messages = [
                {"role": "system", "content": "请根据用户的问题提供回答。"},
                {"role": "user", "content": qa_instruction}
            ]

            response = self.predict(messages, self.model, self.tokenizer)

            result_entry = {
                "question": row["question"],
                "history": row["history"],
                "documents": row["documents"],
                "answer": row["answer"],
                "prediction": response
            }
            results.append(result_entry)

        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Results saved to result.json")

        
        if self.training:
            test_text_list = [
                swanlab.Text(
                    f"Question: {r['question']}\n\nGround Truth: {r['answer']}\n\nModel Answer: {r['prediction']}",
                    caption=r["prediction"]
                ) for r in results[:10]
            ]
            swanlab.log({"Prediction": test_text_list})
        
    def predict(self, messages, model, tokenizer, device="cuda"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id  
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response