import re
from src.rag import make_system_prompt, query_arvind_facts


messages=[]


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

def generate_response(tokenizer, model, messages, temperature=0.9):
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

def chat_with_arvind(tokenizer, model, user_input, temperature=0.9):
  global messages

  formatted_facts = query_arvind_facts(user_input)
  system_prompt = make_system_prompt(formatted_facts)

  messages.append({'role':'user',
                  'content':user_input})

  insert_system_prompt(messages, system_prompt)
  response=generate_response(tokenizer, model, messages)


  messages.append({'role':'assistant',
                  'content':response})
  
  return clean_response(response)

def clear_chat():
  global messages
  messages = []