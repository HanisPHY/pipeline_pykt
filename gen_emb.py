import pandas as pd
from transformers import BertTokenizer, BertModel
import torch


file_path = './data/gkt_new/dataset_cogedu_raw.csv'
df = pd.read_csv(file_path, delimiter='\t')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def generate_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embeddings


combined_embeddings = []

for index, row in df.iterrows():
    question_content = row['question_content']
    course_content = row['course_content']

    combined_text = f"{question_content} {course_content}"

    combined_embedding = generate_bert_embeddings(combined_text, tokenizer, model)
    combined_embeddings.append(','.join(map(str, combined_embedding)))

df['embedding_bert'] = combined_embeddings
df.to_csv('cogedu_emb_raw.csv', index=False)
