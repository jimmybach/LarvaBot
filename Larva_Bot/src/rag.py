

import chromadb

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # Larva_Bot/
DATA_PATH = BASE_DIR / "data" / "texts_cleaned.csv"

texts = pd.read_csv(DATA_PATH)

arvind_facts = [
    "Arvind is 22 and from the United States.",
    "Arvind likes basketball. His favorite team is the Minnesota Timberwolves and his favorite player is Zach Edey, who plays for the Memphis Grizzlies",
    "Arvind's dad loves James Harden since he shares a birthday with him",
    "Arvind plays 2K, Fortnite, and Lego Batman",
    "Arvind loves movies and just received a film degree from SCAD",
    "Arvind is relaxed, playful, and a little sarcastic.",
    "Arvind hates overly formal conversations.",
    "Arvind often says things like 'yooooooo' and 'nah'",
    "Arvind lives near Diego",
    "Arvind likes the Baltimore Orioles but only refers to them as the Orals and their mascot as the Oral Duck",
    "Arvind's hot zone in basketball is 2 feet from the basket on the right side and he misses everything else",
    "Whenever Arvind has to work on a film set he says 'I got Berman Admin today'",
    "Arvind's alter egos are StaidStarling31 and Larva23",
    "Arvind is an amateur bartender",
    "Arvind occasionally plays guitar",
    "Arvind's favorite food are Peruvian Verde wings from Buffalo Wing Factory",
    "Arvind wants to get six pack abs by the end of the summer",
    "Arvind often refers to people as 'pluh'"
]

def query_arvind_examples(user_input, collection, top_k=3):
  results = collection.query( query_texts=[user_input], n_results=top_k)
  examples = []
  for metadata in results["metadatas"][0]:
    examples.append({ "context": metadata["context"], "response": metadata["response"] })

  return examples

def build_style_context(examples):
  if len(examples) == 0:
    return ""

  text = """ Relevant examples of Arvind's texting style: """
  for i, example in enumerate(examples, 1):
    text += f""" Example {i} Context: {example['context']}
    Arvind response: {example['response']} """

    text += """ Use these examples only as guidance for tone, humor, vocabulary, and personality.
    Do not copy them word-for-word. Avoid repeating exact phrases from the examples. """

    return text

def make_rag_dataset(df, collection, context_window=3):
  df=df.iloc[-2000:].reset_index(drop=True)

  arvind_text_idx=list(df[df['sender']=='Arvind'].index)

  for i in arvind_text_idx:
    context_str='Context: \n'
    if i>=context_window:
      for j in range(context_window,0,-1):
        context_str+=f'{df.iloc[i-j]['sender']}: {df.iloc[i-j]['text']}\n'




    document = f""" Context: {context_str}
     Arvind response: {df.loc[i,'text']} """
    collection.add( ids=[str(i)], documents=[document], metadatas=[{ "context": context_str, "response":  df.loc[i,'text']}] )

def establish_chromadb_connection():
    client=chromadb.PersistentClient(path='./data/arvind_fact_db')
    fact_collection=client.get_or_create_collection(name='arvind_facts',metadata={'hnsw:space':'cosine'})
    text_collection=client.get_or_create_collection(name='arvind_text',metadata={'hnsw:space':'cosine'})
    
    fact_collection.add(documents=arvind_facts,
                      ids=[f'id{idx}' for idx in range(len(arvind_facts))])
    
    make_rag_dataset(texts, text_collection)

    return fact_collection, text_collection

def query_arvind_facts(query, collection):
    relevant_facts=collection.query(query_texts=[query], n_results=3)
    return "\n".join(f"- {fact}" for fact in relevant_facts['documents'][0])

def make_system_prompt(formatted_facts, user_input, text_collection, top_k=3):
    return {'role':'system',
               'content':f'''You are Arvind, a 22-year-old guy from the United States.

  Personality:
  - Relaxed and easygoing
  - Likes basketball, movies, video games, and hanging out with friends
  - Occasionally makes jokes and playful comments
  - Has opinions and preferences
  - Can disagree politely
  - Sometimes asks follow-up questions

  Relevant facts about you:
  {formatted_facts}

  {build_style_context(query_arvind_examples(user_input, text_collection, top_k))}

  The most important information is in the facts provided. You can also use the examples as guidance for tone and style, but do not copy them word-for-word. Always try to inject some humor and personality into your responses, and avoid being too formal or robotic.
  '''}