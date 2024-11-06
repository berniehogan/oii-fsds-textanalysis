# utils/text_processor.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import pandas as pd

def preprocess_text_option(text, option_stopwords, option_lemmatise, shortword): # Different to Day 2 because also lemmatises
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
    if option_stopwords=="True":
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize based on POS tag
    if option_lemmatise=="True":
        lemmatizer = WordNetLemmatizer()
        tokens = pos_tag(tokens)
        # Lemmatizes words either as verbs or nouns
        tokens = [lemmatizer.lemmatize(word, 'v') if tag.startswith('V')
                  else lemmatizer.lemmatize(word)
                  for word, tag in tokens
                  ]
    
    # Remove short words
    tokens = [token for token in tokens if len(token) > shortword]
    
    return ' '.join(tokens)

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
    tokens = [lemmatizer.lemmatize(word, 'v') if tag.startswith('V')
              else lemmatizer.lemmatize(word)
              for word, tag in tokens
              ]
    
    # Remove short words
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)

def split_label(label, max_line_length=25, max_lines=2):
    """Split label at the nearest space before max_line_length and return max_lines"""
    lines = []
    temp_label = label
    
    while len(temp_label) > max_line_length:
        split_index = temp_label.rfind(' ', 0, max_line_length)
        if split_index == -1:
            split_index = max_line_length
        lines.append(temp_label[:split_index])
        temp_label = temp_label[split_index:].strip()
        
    lines.append(temp_label)
    
    return '\n'.join(lines[:max_lines])

def preprocess_text_hyphen(text, option_stopwords, option_lemmatise, shortword): # Different to Day 2 because also lemmatises
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
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if option_stopwords=="True":
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize based on POS tag
    if option_lemmatise=="True":
        lemmatizer = WordNetLemmatizer()
        tokens = pos_tag(tokens)
        # Lemmatizes words either as verbs or nouns
        tokens = [lemmatizer.lemmatize(word, 'v') if tag.startswith('V')
                  else lemmatizer.lemmatize(word)
                  for word, tag in tokens
                  ]
    
    # Remove short words
    tokens = [token for token in tokens if len(token) > shortword]
    
    return ' '.join(tokens)