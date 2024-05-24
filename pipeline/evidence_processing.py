from gensim.utils import deaccent
from nltk import pos_tag
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

import re
import time

import pandas as pd
import json

import pickle
import nltk
nltk.download('averaged_perceptron_tagger')

d_evidence = pd.read_json("data/evidence.json", typ='series')

lemmatizer = WordNetLemmatizer()

# contraction_dict from WS7
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                    "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                    "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",
                    "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                    "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",
                    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


# https://stackoverflow.com/a/46231553
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 


def sentence_preprocessing(sentence):

    out_list = []
    # Use gensim deaccent to match more characters to [a-z]
    sentence = deaccent(sentence.lower())

    for old, new in contraction_dict.items():
        sentence.replace(old, new)

    tokenized = word_tokenize(sentence)

    # now remove all tokens that don't contain any alphanumeric characters
    # then strip non alphanumeric characters afterwards
    tokenized = [re.sub(r"[^a-z0-9\s]", "", token) for token in tokenized if re.match(r"[a-z0-9\s]", token)]

    # now lemmatize with pos
    tagged = pos_tag(tokenized)
    for token, tag in tagged:
        wntag = get_wordnet_pos(tag)

        if wntag is None: # do not supply tag in case of None
            lemma = lemmatizer.lemmatize(token) 
        else:
            lemma = lemmatizer.lemmatize(token, pos=wntag) 

        out_list.append(lemma)
    
    return out_list


def evidence_preprocessing(evidences):
  t = time.time()
  processed = []
  for index, item in enumerate(evidences.items()):
    id, evidence = item

    row = []
    
    row.append(id)
    row.append(evidence)

    # break the text into sentences before tokenizing by each sentence
    processed_sentences = [sentence_preprocessing(sentence) for sentence in sent_tokenize(evidence)]
    row.append(processed_sentences)


    # Appending an empty list to populate with embeddings later
    row.append([])

    processed.append(row)

    if (index + 1) % 50000 == 0:
        print(f"{time.time() - t:.2f} - {index+1} rows processed")

  return pd.DataFrame(processed, columns = ["id", "raw evidence", "processed evidence", "embeddings"])

evidence = evidence_preprocessing(d_evidence)
with open("evidence_preprocessed_v3.pkl", "wb") as f:
    pickle.dump(evidence, f)