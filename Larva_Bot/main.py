from unicodedata import name
from src.finetuning import initialize_and_train, merge_and_save
from src.make_chat import chat_with_arvind
from src.make_finetuning_data import make_finetuning_dataset, split_and_save
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

def main():
    data=pd.read_csv("./data/texts_cleaned.csv")
    finetuning_data=make_finetuning_dataset(data)
    split_and_save(finetuning_data)
    print("Finished preparing finetuning data!")
    
    dataset=load_dataset("json", data_files="arvind_train.jsonl")["train"]

    model_name = "Qwen/Qwen3-4B"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Finished loading model and tokenizer!")
    trainer = initialize_and_train(model, tokenizer, dataset)
    print("Finished finetuning!")
    merge_and_save(trainer,tokenizer)

    chat_with_arvind(tokenizer, model, "What do you like to do for fun?")

if __name__=="__main__":    
    main()