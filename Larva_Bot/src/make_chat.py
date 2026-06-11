import re
import torch
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

SHORT_TERM_MEMORY = 8  # last 4 user/assistant exchanges

def generate_response(tokenizer, model, messages, temperature=0.9, device="cpu"):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=100,
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


def chat_with_arvind(tokenizer, model, user_input, fact_collection, text_collection, temperature=0.9, device="cpu"):
    global messages

    recent_messages = messages[-SHORT_TERM_MEMORY:]

    format_chat_history = lambda msgs: "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in msgs]
    )

    rag_query = format_chat_history(recent_messages) + "\nUser: " + user_input

    formatted_facts = query_arvind_facts(rag_query, fact_collection)

    system_prompt = make_system_prompt(
        formatted_facts,
        user_input,
        text_collection
    )

    generation_messages = [
        {"role": "system", "content": system_prompt},
        *recent_messages,
        {"role": "user", "content": user_input}
    ]

    response = generate_response(
        tokenizer,
        model,
        generation_messages,
        device=device,
        temperature=temperature
    )

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response})

    if len(messages) > SHORT_TERM_MEMORY:
        messages = messages[-SHORT_TERM_MEMORY:]

    return clean_response(response)


def clear_chat():
    global messages
    messages = []