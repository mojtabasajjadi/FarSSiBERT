# -*- coding: utf-8 -*-
"""
@author: Seyed Mojtaba Sadjadi
"""

import re
import hazm
from cleantext import clean
import torch
from transformers import BertTokenizer
from transformers import BertModel
from sentence_transformers import util
from importlib import resources
import io
import os

# Current directory
here = os.path.abspath(os.path.dirname(__file__))

# Load model 
model = BertModel.from_pretrained(here, local_files_only=True)    
tokenizer = BertTokenizer.from_pretrained(here + '/trainedTokenizer-vocab.txt', local_files_only=True)
    
class SSMeasurement:
    
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
        
    # Pre-processing methods 
    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
    
    def cleaning(text):
        text = text.strip()
        # regular cleaning
        text = clean(text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            replace_with_url="URL",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol="",
        )
        # cleaning htmls
        text = SSMeasurement.cleanhtml(text)
        # normalizing
        normalizer = hazm.Normalizer()
        text = normalizer.normalize(text)
        # removing wierd patterns
        wierd_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u'\U00010000-\U0010ffff'
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            u"\u2069"
            u"\u2066"
            # u"\u200c"
            u"\u2068"
            u"\u2067"
            "]+", flags=re.UNICODE)
        text = wierd_pattern.sub(r'', text)
        # removing extra spaces, hashtags
        text = re.sub("#", "", text)
        text = re.sub("\s+", " ", text)
        return text
    
    # Pooling methods 
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Max Pooling - Take the max value over time for every dimension. 
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]


    # Get sentences' similarity label
    def get_similarity_label(self):
        tweet1 = self.text1
        tweet2 = self.text2
        sentences = [SSMeasurement.cleaning(str(tweet1)), SSMeasurement.cleaning(str(tweet2))]
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = SSMeasurement.mean_pooling(model_output, encoded_input['attention_mask'])
        tweet1_emb = sentence_embeddings[0,:].flatten().tolist()
        tweet2_emb = sentence_embeddings[1,:].flatten().tolist()
        #    print("Sentence embeddings:")
        #    print(sentence_embeddings)
        label = 0
        cosine_scores = util.cos_sim(tweet1_emb, tweet2_emb)
        if 0 < cosine_scores.item() <= 0.69:
            label = 0
        elif 0.69 < cosine_scores.item() <= 0.75:
            label =1
        elif 0.75 < cosine_scores.item() <= 0.80:
            label =2
        elif 0.80 < cosine_scores.item() <= 0.88:
            label =3
        elif 0.88 < cosine_scores.item() <= 0.95:
            label =4
        elif 0.95 < cosine_scores.item():
            label =5
        return label
      
    # Get cosine similarity of sentences
    def get_cosine_similarity(self):
        tweet1 = self.text1
        tweet2 = self.text2
        sentences = [SSMeasurement.cleaning(str(tweet1)), SSMeasurement.cleaning(str(tweet2))]
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = SSMeasurement.mean_pooling(model_output, encoded_input['attention_mask'])
        tweet1_emb = sentence_embeddings[0,:].flatten().tolist()
        tweet2_emb = sentence_embeddings[1,:].flatten().tolist()
        cosine_scores = util.cos_sim(tweet1_emb, tweet2_emb)
#        print(cosine_scores.item())
        return (cosine_scores.item())



