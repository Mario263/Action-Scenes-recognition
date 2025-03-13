import json
from transformers import LlavaProcessor, LlavaForConditionalGeneration, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from trl import SFTTrainer
import os

# 🚀 Fix 1: Force CPU execution to avoid MPS out-of-memory errors
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # Prevent MPS fallback
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ensure CUDA is off
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Prevent MPS memory issues
os.environ["TORCH_MPS_DISABLE"] = "1" 

# 🚀 Fix 2: Limit MPS memory usage & clear cache
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(0.8)  # Use max 80% VRAM
    torch.mps.empty_cache()  # Free unused memory

# Load model and processor without CUDA or 4-bit quantization
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)  # Use fp32 for CPU

# Move model to device
model.to(device)

# 🚀 Fix 3: Ensure proper weight initialization to prevent NaNs
for name, param in model.named_parameters():
    if param.requires_grad and param.dim() > 1:  # Apply only to weight matrices
        torch.nn.init.xavier_uniform_(param)

# Data Loading
def read_json_file(filename="/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/limited_processed_data.json"):
    """Read JSON file and yield entries."""
    with open(filename, 'r') as f:
        data = json.load(f)
        for entry in data:
            yield entry

# Collate function
def collate_fn(batch):
    batch_texts = []
    batch_images = []

    for entry in batch:
        batch_texts.append(entry["text"])
        for image in entry["frames"]:
            batch_images.append(Image.open(image))

    processed_data = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )

    labels = processed_data["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    processed_data["labels"] = labels
    return processed_data

# Add text column
def add_text(data):
    data["text"] = data["prompt"]
    return data

if __name__ == "__main__":
    dataset = load_dataset("json", data_files="/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/limited_processed_data.json", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(add_text)
    train_test_split = dataset.train_test_split(test_size=0.2)

    # LoRA Configuration
    target_modules = ["q_proj", "v_proj", "fc1", "fc2"]
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    # Apply LoRA after initializing model weights
    peft_model = get_peft_model(model, peft_config).to(device)

    print("Trainable Parameters:")
    peft_model.print_trainable_parameters()

    # 🚀 Fix 4: Reduce batch size & gradient accumulation for M1 Mac
    print("Setting up Training Arguments")
    training_args = TrainingArguments(
        output_dir='/Users/mario/Desktop/Desktop/UofA/4.Winter-2025/ece-910/Batman/VLM-testing/VLM-output/runs',
        num_train_epochs=2,  # Increase epochs to compensate for smaller batches
        gradient_accumulation_steps=1,  # Reduced from 8 to 4 to prevent memory overflow
        per_device_train_batch_size=1,  # Small batch size to fit into memory
        per_device_eval_batch_size=1,
        warmup_steps=100,  # Higher warmup to prevent NaN gradients
        weight_decay=0.01,
        evaluation_strategy='steps',
        eval_steps=5,
        logging_steps=1,
        logging_strategy="steps",
        gradient_checkpointing=True,
        save_steps=500,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    print("Starting Training...")
    trainer.train()