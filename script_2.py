
# Create comprehensive data preprocessing Python notebook/script
# This will include tokenization, lemmatization, and stopword removal

preprocessing_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Customer Feedback Analysis - Data Preprocessing
================================================
This script performs comprehensive data cleaning and preprocessing for customer feedback analysis.

Author: AI Customer Feedback Analysis System
Date: October 2025
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class FeedbackPreprocessor:
    """
    A comprehensive preprocessing class for customer feedback data
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def remove_duplicates(self, df):
        """Remove duplicate rows from the dataset"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        print(f"✓ Removed {removed} duplicate rows")
        return df
    
    def handle_missing_data(self, df, column='feedback_text'):
        """Handle missing values in the dataset"""
        initial_rows = len(df)
        missing_count = df[column].isnull().sum()
        df = df.dropna(subset=[column])
        print(f"✓ Removed {missing_count} rows with missing {column}")
        return df
    
    def clean_special_characters(self, text):
        """Remove special characters while preserving sentence structure"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\\S+@\\S+', '', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\\s\\.\\,\\!\\?]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([!?.]){2,}', r'\\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        if not text or pd.isna(text):
            return []
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their base form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline for a single text"""
        # Clean special characters
        cleaned = self.clean_special_characters(text)
        
        # Tokenization
        tokens = self.tokenize_text(cleaned)
        
        # Remove punctuation tokens
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatization
        tokens = self.lemmatize_tokens(tokens)
        
        # Remove single characters
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess_dataset(self, input_file, output_file):
        """
        Complete preprocessing pipeline for the entire dataset
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        output_file : str
            Path to save cleaned CSV file
        """
        print("="*60)
        print("CUSTOMER FEEDBACK DATA PREPROCESSING")
        print("="*60)
        
        # Load data
        print(f"\\n1. Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"   Initial dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Remove duplicates
        print("\\n2. Removing duplicates...")
        df = self.remove_duplicates(df)
        
        # Handle missing data
        print("\\n3. Handling missing data...")
        df = self.handle_missing_data(df, 'feedback_text')
        
        # Clean text
        print("\\n4. Cleaning special characters...")
        df['feedback_text_cleaned'] = df['feedback_text'].apply(self.clean_special_characters)
        
        # Tokenization
        print("\\n5. Tokenizing text...")
        df['tokens'] = df['feedback_text_cleaned'].apply(self.tokenize_text)
        
        # Remove stopwords
        print("\\n6. Removing stopwords...")
        df['tokens_no_stopwords'] = df['tokens'].apply(self.remove_stopwords)
        
        # Lemmatization
        print("\\n7. Lemmatizing tokens...")
        df['tokens_lemmatized'] = df['tokens_no_stopwords'].apply(self.lemmatize_tokens)
        
        # Create processed text (join tokens back)
        df['feedback_processed'] = df['tokens_lemmatized'].apply(lambda x: ' '.join(x))
        
        # Save processed dataset
        print(f"\\n8. Saving processed dataset to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print("\\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\\nSample of processed feedback:")
        print("-"*60)
        for idx in df.head(3).index:
            print(f"\\nOriginal: {df.loc[idx, 'feedback_text'][:80]}...")
            print(f"Processed: {df.loc[idx, 'feedback_processed'][:80]}...")
        
        print("\\n" + "="*60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return df

# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = FeedbackPreprocessor()
    
    # Process the dataset
    df_processed = preprocessor.preprocess_dataset(
        input_file='customer_feedback_raw.csv',
        output_file='customer_feedback_preprocessed.csv'
    )
    
    print("\\n✓ You can now use 'customer_feedback_preprocessed.csv' for model training!")
'''

# Save the preprocessing script
with open('data_preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(preprocessing_code)

print("✓ Created 'data_preprocessing.py' script")
print("\nScript includes:")
print("  - Duplicate removal")
print("  - Missing data handling")
print("  - Special character cleaning")
print("  - Tokenization")
print("  - Stopword removal")
print("  - Lemmatization")
print("\nThe script is ready to run!")
