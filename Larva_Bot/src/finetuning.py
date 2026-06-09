from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir="./arvind-qwen-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=200,
    logging_steps=10,
    save_steps=100,
    bf16=False,
    fp16=False,
    report_to="none",
    dataloader_pin_memory=False
)

def initialize_and_train(model, tokenizer,dataset):
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer
    )

    trainer.train()
    return trainer

def merge_and_save(trainer,tokenizer):
    merged_model = trainer.model.merge_and_unload()

    merged_model.save_pretrained("./models/arvind-merged")
    tokenizer.save_pretrained("./models/arvind-merged")