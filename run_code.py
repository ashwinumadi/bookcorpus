from datasets import load_dataset

# Load the BookCorpus dataset
dataset = load_dataset("bookcorpus", trust_remote_code=True)

from collections import Counter
from tqdm import tqdm
import pickle

# Initialize a Counter to keep track of occurrences
#string_count = Counter()

# Load the Counter object from the pickle file
with open('ufet_count2.pkl', 'rb') as f:
    string_count = pickle.load(f)

# Access the training data
train_data = dataset['train']

# View the first entry in the training data
print(train_data[0])

import json
data = []
with open('/content/release/crowd/train.json', 'r') as file:
  for line in file:
    data.append(json.loads(line))
with open('/content/release/crowd/test.json', 'r') as file:
  for line in file:
    data.append(json.loads(line))
with open('/content/release/crowd/dev.json', 'r') as file:
  for line in file:
    data.append(json.loads(line))
unique_entities = set()
for obj in data:
  unique_entities.add(obj['mention_span'])

unique_entities = {word.lower() for word in unique_entities}

a = train_data[5000000:40000000]
# Iterate over the dataset and count occurrences of strings in your set
for entry in tqdm(a['text'], desc="Processing items", ncols=100):
    text = entry  # Assuming the text is in the 'text' field, modify if needed
    for word in unique_entities:
        string_count[word] += text.lower().count(word.lower())

# Display the count of each string in the set


# Store the Counter object in a pickle file
with open('ufet_count3.pkl', 'wb') as f:
    pickle.dump(string_count, f)
