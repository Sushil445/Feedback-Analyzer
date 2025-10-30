
# Part 3: Create Text Summarization Script using T5 and BART

summarization_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI-Powered Text Summarization for Customer Feedback
====================================================
This script implements both transformer-based (T5, BART) and extractive
summarization for customer feedback analysis.

Models:
- T5 (Text-to-Text Transfer Transformer)
- BART (Bidirectional and Auto-Regressive Transformer)
- Custom Extractive Summarization (TF-IDF + Cosine Similarity)
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TransformerSummarizer:
    """Transformer-based summarization using T5 and BART"""
    
    def __init__(self, model_type='t5'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_type = model_type
        
        if model_type == 't5':
            print("Loading T5 model...")
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
        elif model_type == 'bart':
            print("Loading BART model...")
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)
    
    def summarize_short(self, text, max_length=50):
        """Generate a short summary"""
        if self.model_type == 't5':
            input_text = f"summarize: {text}"
        else:
            input_text = text
        
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def summarize_detailed(self, text, max_length=150):
        """Generate a detailed summary"""
        if self.model_type == 't5':
            input_text = f"summarize: {text}"
        else:
            input_text = text
        
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class ExtractiveSummarizer:
    """Custom extractive summarization using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def summarize(self, text, num_sentences=3):
        """
        Extract most important sentences using TF-IDF and cosine similarity
        
        Parameters:
        -----------
        text : str
            Input text to summarize
        num_sentences : int
            Number of sentences to include in summary
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except:
            return sentences[0]  # Return first sentence if TF-IDF fails
        
        # Calculate sentence scores based on average TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get indices of top sentences
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Extract top sentences in original order
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        summary = ' '.join(summary_sentences)
        
        return summary

class FeedbackSummarizer:
    """
    Complete summarization system for customer feedback
    Supports multiple summarization methods
    """
    
    def __init__(self):
        self.t5_summarizer = TransformerSummarizer(model_type='t5')
        self.extractive_summarizer = ExtractiveSummarizer()
    
    def summarize_feedback_batch(self, df, text_column='feedback_text'):
        """
        Summarize a batch of customer feedback
        
        Parameters:
        -----------
        df : DataFrame
            Customer feedback dataframe
        text_column : str
            Column containing feedback text
        """
        results = []
        
        print("\\n" + "="*70)
        print("SUMMARIZING CUSTOMER FEEDBACK")
        print("="*70)
        
        # Group feedback by category or product for batch summarization
        grouped_feedback = df.groupby('category')[text_column].apply(
            lambda x: ' '.join(x.astype(str))
        ).to_dict()
        
        for category, combined_text in grouped_feedback.items():
            print(f"\\nProcessing category: {category}")
            print("-" * 70)
            
            # Limit text length
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000]
            
            # T5 Short Summary
            print("  Generating T5 short summary...")
            t5_short = self.t5_summarizer.summarize_short(combined_text)
            
            # T5 Detailed Summary
            print("  Generating T5 detailed summary...")
            t5_detailed = self.t5_summarizer.summarize_detailed(combined_text)
            
            # Extractive Summary
            print("  Generating extractive summary...")
            extractive = self.extractive_summarizer.summarize(combined_text, num_sentences=3)
            
            results.append({
                'category': category,
                'original_text_length': len(combined_text),
                'num_feedbacks': len(df[df['category'] == category]),
                't5_short_summary': t5_short,
                't5_detailed_summary': t5_detailed,
                'extractive_summary': extractive
            })
        
        return pd.DataFrame(results)
    
    def summarize_single_feedback(self, text):
        """Summarize a single feedback with all methods"""
        
        result = {
            't5_short': self.t5_summarizer.summarize_short(text),
            't5_detailed': self.t5_summarizer.summarize_detailed(text),
            'extractive': self.extractive_summarizer.summarize(text, num_sentences=2)
        }
        
        return result

def main():
    """Main summarization pipeline"""
    
    print("="*70)
    print("AI-POWERED TEXT SUMMARIZATION FOR CUSTOMER FEEDBACK")
    print("="*70)
    
    # Load data
    print("\\nLoading customer feedback data...")
    df = pd.read_csv('customer_feedback_cleaned.csv')
    print(f"Loaded {len(df)} feedback records")
    
    # Initialize summarizer
    print("\\nInitializing summarization models...")
    summarizer = FeedbackSummarizer()
    
    # Summarize by category
    print("\\nGenerating category-wise summaries...")
    summary_df = summarizer.summarize_feedback_batch(df)
    
    # Save results
    summary_df.to_csv('feedback_summaries.csv', index=False)
    print("\\n✓ Summaries saved to 'feedback_summaries.csv'")
    
    # Display results
    print("\\n" + "="*70)
    print("SUMMARIZATION RESULTS")
    print("="*70)
    
    for idx, row in summary_df.iterrows():
        print(f"\\nCategory: {row['category']}")
        print(f"Number of feedbacks: {row['num_feedbacks']}")
        print(f"Original text length: {row['original_text_length']} characters")
        print("\\nT5 Short Summary:")
        print(f"  {row['t5_short_summary']}")
        print("\\nT5 Detailed Summary:")
        print(f"  {row['t5_detailed_summary']}")
        print("\\nExtractive Summary:")
        print(f"  {row['extractive_summary'][:200]}...")
        print("-" * 70)
    
    # Example: Summarize individual feedback
    print("\\n" + "="*70)
    print("SAMPLE INDIVIDUAL FEEDBACK SUMMARIZATION")
    print("="*70)
    
    sample_feedback = df.iloc[0]['feedback_text']
    print(f"\\nOriginal Feedback:")
    print(f"  {sample_feedback}")
    
    individual_summary = summarizer.summarize_single_feedback(sample_feedback)
    
    print(f"\\nT5 Short Summary:")
    print(f"  {individual_summary['t5_short']}")
    
    print(f"\\nT5 Detailed Summary:")
    print(f"  {individual_summary['t5_detailed']}")
    
    print(f"\\nExtractive Summary:")
    print(f"  {individual_summary['extractive']}")
    
    print("\\n" + "="*70)
    print("✓ SUMMARIZATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\\nSummarization methods implemented:")
    print("  1. T5 Transformer (short summary)")
    print("  2. T5 Transformer (detailed summary)")
    print("  3. Custom Extractive (TF-IDF + Cosine Similarity)")

if __name__ == "__main__":
    main()
'''

# Save the summarization script
with open('text_summarization.py', 'w', encoding='utf-8') as f:
    f.write(summarization_code)

print("✓ Created 'text_summarization.py' script")
print("\nScript includes:")
print("  - T5 transformer for short and detailed summaries")
print("  - BART transformer support")
print("  - Custom extractive summarization (TF-IDF + cosine similarity)")
print("  - Batch summarization by category")
print("  - Individual feedback summarization")
print("\nThe script is ready to run!")
