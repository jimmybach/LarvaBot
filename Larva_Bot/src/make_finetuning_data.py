import pandas as pd
from sklearn.model_selection import train_test_split as tts
import json

def make_finetuning_dataset(df, context_window=5):
  finetuning_data=[]
  df=df.reset_index(drop=True)
  arvind_text_idx=list(df[df['sender']=='Arvind'].index)

  system_prompt={'role': 'system',
                 'content': 'You are Arvind, a 22 year old man who texts very casually'}


  for i in arvind_text_idx:
    context_str=''
    if i>=context_window:
      for j in range(context_window,0,-1):
        context_str+=f'{df.iloc[i-j]['sender']}: {df.iloc[i-j]['text']}\n'
    context_prompt={'role':'user',
                    'content':context_str}

    assistant_prompt={'role':'assistant',
                      'content':df.iloc[i]['text']}

    finetuning_data.append({'messages':[system_prompt,context_prompt,assistant_prompt]})
  return finetuning_data

def split_and_save(finetuning_data):

    train, test = tts(
        finetuning_data,
        test_size=0.2,
        random_state=42
    )
    
    def save_jsonl(examples, path):
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    save_jsonl(train, "arvind_train.jsonl")
    save_jsonl(test, "arvind_test.jsonl")


