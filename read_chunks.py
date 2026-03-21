import requests
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib   

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "mxbai-embed-large",
        "input": text_list
    })

    data = r.json()

    if "embeddings" not in data:
        print("Error response:", data)
        return []

    return data["embeddings"]

jsons = os.listdir("jsons")

my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}") 

    texts = [c['text'] for c in content.get('chunks', [])]
    embeddings = create_embedding(texts)

    for i, chunk in enumerate(content.get("chunks", [])):
        chunk['chunk_id'] = chunk_id

        if i < len(embeddings):
            chunk['embedding'] = embeddings[i]
        else:
            chunk['embedding'] = None  # safety fallback

        chunk_id += 1
        my_dict.append(chunk)
        
    

   

df = pd.DataFrame.from_records(my_dict)
# print(df)
joblib.dump(df, "embeddings.joblib")
incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)


#find similarity of question_embedding with other embeddings 
similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_results = 3
max_indx =  similarities.argsort()[::-1][0:top_results]
print(max_indx)
new_df = df.iloc[max_indx]
print(new_df[["title", "number", "text"]])
