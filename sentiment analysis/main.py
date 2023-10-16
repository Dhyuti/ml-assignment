
import pandas as pd
import os

# Check the available files in the specified directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read the dataset
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'IMDB Dataset.csv')

df = pd.read_csv(file_path)

# Display the first few rows, shape, and information of the DataFrame
print(df.head())
print(df.shape)
df.info()

# Reset the index and rename the 'index' column to 'Id'
df = df.reset_index(drop=False)
df = df.rename(columns={'index': 'Id'})

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the BERT-based sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Define a function to calculate sentiment scores
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

print(sentiment_score('I hate you'))

# Apply the sentiment_score function to the 'review' column and store the results in a new 'score' column
df['score'] = df['review'].apply(lambda x: sentiment_score(x[:512]))