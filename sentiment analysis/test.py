from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd

# Load the BERT-based sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Define a function to calculate sentiment scores
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Test the function
print(sentiment_score('I hate you'))
print(sentiment_score('I love you'))

# Read the dataset
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'IMDB Dataset.csv')

df = pd.read_csv(file_path)
df['score'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
