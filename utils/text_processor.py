# utils/text_processor.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import pandas as pd

def preprocess_text(text): # Different to Day 2 because also lemmatises
    """
    Clean and normalize text using NLTK.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize based on POS tag
    lemmatizer = WordNetLemmatizer()
    tokens = pos_tag(tokens)
    # Lemmatizes words either as verbs or nouns
    tokens = [
        lemmatizer.lemmatize(word, 'v') if tag.startswith('V')
        else lemmatizer.lemmatize(word)
        for word, tag in tokens
    ]
    
    # Remove short words
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)
