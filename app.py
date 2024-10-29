import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import os
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import re
from collections import Counter

#setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import re

#path of the RTF file
file_path = "Emma_by_Jane_Austen.rtf"

#reading the RTF file
with open(file_path, 'r', encoding='utf-8') as file:
    rtf_content = file.read()

#function to clean RTF file
def clean_rtf(rtf):
    # Remove RTF formatting
    # This regex removes everything that isn't plain text
    cleaned_text = re.sub(r'{\\.*?}', '', rtf)  # Remove RTF groups
    cleaned_text = re.sub(r'\\[a-z]+\d* ?', '', cleaned_text)  # Remove RTF commands
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    return cleaned_text.strip()

#cleaning the RTF text
plain_text = clean_rtf(rtf_content)

#displaying the first 500 characters of the plain text
#print(plain_text[:500])

#converting text to lowercase
plain_text = plain_text.lower()

#removing unwanted characters
cleaned_text = re.sub('[^a-zA-Z0-9 .]', '', plain_text)

#splitting into words
words = cleaned_text.split()

#print(cleaned_text[:400])

#creating vocab of unique words
words_vocab = sorted(set(words))
stoi = {s: i for i, s in enumerate(words_vocab)}
itos = {i: s for i, s in enumerate(words_vocab)}



class NextWordMLP(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.activation = activation
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.lin2(x)
        return x

def generate_text(model, itos, stoi, block_size, max_length=50):
    context = [0] * block_size
    generated_words = []
    for _ in range(max_length):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = itos[ix]
        generated_words.append(word)
        context = context[1:] + [ix]
    return ' '.join(generated_words)



# Load model function
def load_model(embedding_size, block_size, activation_fn_name):
    model = NextWordMLP(block_size, len(stoi), embedding_size, 512, activation_fn_name).to(device)
    model_filename = f"saved_models/model_emb{embedding_size}_ctx{block_size}_act{activation_fn_name}.pt"
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    return model

# Streamlit App
st.title("Next-Word Prediction App")
st.write("This app predicts the next words based on an input text. Choose a model and configuration, then enter some starting text.")

# Select Model Configuration
embedding_sizes = [64, 128]
context_lengths = [5, 10, 15]
activations = ["relu", "tanh"]

model_options = [
    f"Embedding Size: {emb}, Context Length: {ctx}, Activation: {act}"
    for emb in embedding_sizes
    for ctx in context_lengths
    for act in activations
]

model_selection = st.selectbox("Choose a Model", model_options)

# Parse selected model options
embedding_size, block_size, activation_fn_name = model_selection.split(", ")
embedding_size = int(embedding_size.split(": ")[1])
block_size = int(block_size.split(": ")[1])
activation_fn_name = activation_fn_name.split(": ")[1]

# Load the selected model
model = load_model(embedding_size, block_size, activation_fn_name)

# Enter Starting Text
starting_text = st.text_input("Input some text to start prediction:")
max_length = st.slider("Select max length of generated text:", min_value=5, max_value=100, value=50)

if st.button("Generate Text"):
    if starting_text:
        context = [stoi[word] for word in starting_text.split()[-block_size:]]  # Take the last `block_size` words
        generated_text = generate_text(model, itos, stoi, block_size, max_length=max_length)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter some starting text.")
