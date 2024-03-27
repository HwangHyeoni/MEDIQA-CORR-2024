# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Optional
from transformers import set_seed
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator
from utils import create_and_prepare_model
from tqdm.auto import tqdm
from data_utils import example_utils, jsonl_utils
llama2_prompt = "HUMAN: You are an expert tasked with providing a logical explanation as to whether there is an error in the given clinical note. Your job is to analyze the clinical note step-by-step and provide an explanation leading to the conclusion regarding the presence of absence of an error. You are strongly recommended to follow the output format: \nAt the end of your response, without modifications, use the phrase \"Therefore, the error sentence {put_error_sentence_here} should be corrected to the corrected sentence {put_corrected_sentence_here}.\" or \"Therefore, the note does not contain an error.\n\n[QUESTION]\n\nASSISTANT:"
alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following question completely and accurately.\n\n### Input:\n[QUESTION]\n\n### Response:"
vicuna_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: [QUESTION]\nASSISTANT:"

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The preference dataset to use."},
    )
    train_file: Optional[str] = field(
        default="/data/hyeon/mediqa/cot/test.json", 
        metadata={"help": "train file"}
    )
    validation_file: Optional[str] = field(
        default="/data/hyeon/mediqa/cot/validation.json", 
        metadata={"help": "validation file"}
    )
    test_file: Optional[str] = field(
         default="/data/hyeon/mediqa/cot/validation.json", 
         metadata={"help": "test file"}
    )
    seq_length: Optional[int] = field(
        default=1024, 
        metadata={"help": "Input sequence length"}
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=4096)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"### Question: {example['question']}\n### Answer: {example['answer']}"
    text = llama2_prompt.replace("[QUESTION]", example["text"]) + f"\n{example['cot']}</s>"
    return text

def create_datasets(tokenizer, script_args):
    data_files = {}
    if script_args.train_file is not None:
        data_files["train"] = script_args.train_file
        extension = script_args.train_file.split(".")[-1]
    if script_args.validation_file is not None:
        data_files["validation"] = script_args.validation_file
        val_filename = script_args.validation_file
        extension = script_args.validation_file.split(".")[-1]
    if script_args.test_file is not None:
        data_files["test"] = script_args.test_file
        extension = script_args.test_file.split(".")[-1]                                                                            
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir="datasets")

    print(f"Size of the train set: {len(raw_datasets['train'])}. Size of the validation set: {len(raw_datasets['validation'])}")
    chars_per_token = chars_token_ratio(raw_datasets['train'], tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    # TODO: Do we really need ConstantLengthDataset?
                                                                        
    train_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['train'],
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
        )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['validation'],
        formatting_func=prepare_sample_text,        
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
        )
    test_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['test'],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,   
        chars_per_token=chars_per_token,
        )
    return train_dataset, valid_dataset, test_dataset

def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    # model
    accelerator = Accelerator()
    model, peft_config, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )
    model.to(accelerator.device)
    print("Setting EOS, BOS, and UNK tokens")
    tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
    )
    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.use_unsloth
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # datasets

    validation_examples = jsonl_utils.read_jsonl_2(data_args.validation_file)
    answered_examples = []

# Assuming validation_examples is a list of examples and you have a defined batch size
    batch_size = 16  # Adjust based on your memory constraints
# Helper function to create batches
    def create_batches(examples, batch_size):
        for i in range(0, len(examples), batch_size):
            yield examples[i:i+batch_size]

    tokenizer.padding_side='left'
# Process the validation examples in batches
    with accelerator.split_between_processes(validation_examples) as exs:
        for batch in tqdm(create_batches(exs, batch_size)):
        # Join the texts for each example in the batch to form the input prompts
            batch_prompts = [llama2_prompt.replace("[QUESTION]", ex["text"]) for ex in batch]
        # Tokenize the batch of prompts
            input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(accelerator.device)
            # Generate predictions for the batch
            outputs = model.generate(**input_ids, max_new_tokens=4096, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
        # Decode the generated outputs
            predicted_ans = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        
        # Assign predictions to the examples in the batch
            for i, ex in enumerate(batch):
                ex["prediction"] = predicted_ans[i][len(batch_prompts[i]):]
                answered_examples.append(ex)

    gathered_examples = accelerator.gather_for_metrics(answered_examples)

# Only the main process proceeds to save the predictions
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        jsonl_utils.write_jsonl(os.path.join(training_args.output_dir, "test_predictions.jsonl"), gathered_examples)



    #jsonl_utils.write_jsonl(os.path.join(training_args.output_dir, "test_predictions.jsonl"), validation_examples)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
