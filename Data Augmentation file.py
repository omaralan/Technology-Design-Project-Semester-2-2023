# -*- coding: utf-8 -*-
"""With Aug.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f_7108PxeZN9O6IskXZrGNZZ0OwHWa_B
"""

import pandas as pd
import nltk

import random
from nltk.corpus import stopwords, wordnet

# Download the necessary datasets
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords
stop_words = set(stopwords.words('english'))

import os
import re

# Upload CSV File
from google.colab import files
uploaded = files.upload()

# Loading CSV File
df = pd.read_csv("Merged Data.csv")

df.head()

# Re-confirm the distribution of labels in the df DataFrame
print(df['Result'].value_counts())

"""Data Augmentation"""

# Function to get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Adjusted function for synonym replacement using WordNet
def synonym_replacement_wordnet(text, n=5):
    words = text.split()
    new_words = words.copy()
    num_replaced = 0

    for word in words:
        if word in stop_words:
            continue

        synonyms = get_synonyms(word)

        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)  # Randomly choose a synonym
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1

        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    return sentence

from nltk.sem.drt import DrtFunctionVariableExpression
# Function for random deletion
def random_deletion(sentence, p=0.5):
    words = sentence.split()
    if len(words) == 1:
        return words[0]
    remaining_words = [word for word in words if random.uniform(0, 1) > p]
    if len(remaining_words) == 0:
        return random.choice(words)
    return ' '.join(remaining_words)

# Function for random swap
def random_swap(sentence, n=5):
    words = sentence.split()
    length = len(words)
    if length < 2:
        return sentence
    for _ in range(n):
        idx1, idx2 = random.randint(0, length-1), random.randint(0, length-1)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

# Function to apply all the augmentation techniques on the dataset
# Redefining the augment_data_for_no function to print intermediate results for diagnosis

def augment_data_for_no(data):
    # Separate the data into 'yes' and 'no' subsets
    yes_df = data[data['Result'] == 'yes']
    no_df = data[data['Result'] == 'no']

    print(f"Number of 'no' instances before augmentation: {len(no_df)}")

    augmented_data = []

    # Only augment the 'no' subset
    for index, row in no_df.iterrows():
        # Original data
        augmented_data.append((row['Cleaned Text'], row['Result']))

        # Synonym Replacement
        augmented_data.append((synonym_replacement_wordnet(row['Cleaned Text']), row['Result']))

        # Random Deletion
        augmented_data.append((random_deletion(row['Cleaned Text']), row['Result']))

        # Random Swap
        augmented_data.append((random_swap(row['Cleaned Text']), row['Result']))

    # Convert to DataFrame
    augmented_no_df = pd.DataFrame(augmented_data, columns=['Cleaned Text', 'Result'])

    print(f"Number of 'no' instances after augmentation: {len(augmented_no_df)}")

    # Combine the augmented 'no' data with the original 'yes' data
    combined_df = pd.concat([yes_df, augmented_no_df], ignore_index=True)

    print(f"Total number of instances after combining 'yes' and augmented 'no': {len(combined_df)}")

    return combined_df

# Apply Data Augmentation
augmented_df = augment_data_for_no(df)

print(augmented_df['Result'].value_counts())

augmented_df.shape

from google.colab import files

# Save the augmented_df DataFrame to a CSV file
augmented_df.to_csv('augmented_data.csv', index=False)

# Trigger a download of the CSV file to your local machine
files.download('augmented_data.csv')