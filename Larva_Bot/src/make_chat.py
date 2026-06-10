import re
import torch
from src.rag import make_system_prompt, query_arvind_facts


messages=[]

few_shot_examples=[
    {"role":"user",
     "content":"What's your biggest regret?"},
    {"role":"assistant",
     "content":"That I'm not Diego :("},
    {"role":"user",
     "content":"Yo Arvind what are you doing today"},
    {"role":"assistant",
     "content":"Pluh I got Berman Admin"},
    {"role":"user",
     "content":"What's your favorite food?"},
    {"role":"assistant",
     "content":"Peruvian verde wings from bwf are the best ngl"},
]

def insert_system_prompt(messages, system_prompt):
  if len(messages)>0:

    if messages[0]['role']!='system':
      messages.insert(0,system_prompt)

    else:
      messages[0]=system_prompt

  else:
    messages.append(system_prompt)

def clean_response(text):
    return re.sub(r"[^\w\s]", "", text).capitalize()

def generate_response(tokenizer, model, messages, temperature=0.9, device="cpu"):
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

  inputs = tokenizer(prompt, return_tensors="pt").to(device)

  with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

  response = tokenizer.decode(
      generated[0][inputs["input_ids"].shape[-1]:],
      skip_special_tokens=True
  )

  return response

def chat_with_arvind(tokenizer, model, user_input, collection, temperature=0.9, device="cpu"):
  global messages

  formatted_facts = query_arvind_facts(user_input, collection)
  system_prompt = make_system_prompt(formatted_facts)
  insert_system_prompt(messages, system_prompt)

  if len(messages)==1:
    messages+=few_shot_examples

  messages.append({'role':'user',
                  'content':user_input})


  response=generate_response(tokenizer, model, messages, device=device, temperature=temperature)


  messages.append({'role':'assistant',
                  'content':response})
  
  if len(messages)>11:
    messages=[messages[0]]+messages[-10:]
  
  return clean_response(response)

def clear_chat():
  global messages
  messages = []