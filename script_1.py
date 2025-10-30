
# Part 1 Continued: Data Preprocessing Script

import pandas as pd
import numpy as np
import re
import string
from collections import Counter

# Load raw data
df = pd.read_csv('customer_feedback_raw.csv')

print("="*50)
print("STEP 1: Initial Data Inspection")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# Add some missing values and duplicates for realistic preprocessing
df_with_issues = df.copy()

# Introduce missing values (5% random)
missing_indices = np.random.choice(df_with_issues.index, size=int(0.05 * len(df_with_issues)), replace=False)
df_with_issues.loc[missing_indices, 'feedback_text'] = np.nan

# Add duplicates
duplicate_rows = df_with_issues.sample(n=20)
df_with_issues = pd.concat([df_with_issues, duplicate_rows], ignore_index=True)

# Add some noise to feedback text
for idx in df_with_issues.sample(n=50).index:
    if pd.notna(df_with_issues.loc[idx, 'feedback_text']):
        df_with_issues.loc[idx, 'feedback_text'] += " !!!@@@###"

# Save data with issues
df_with_issues.to_csv('customer_feedback_with_issues.csv', index=False)

print(f"\nAfter introducing issues:")
print(f"Shape with duplicates and missing values: {df_with_issues.shape}")
print(f"Missing values: {df_with_issues['feedback_text'].isnull().sum()}")
print(f"Duplicate rows: {df_with_issues.duplicated().sum()}")

print("\n" + "="*50)
print("STEP 2: Data Cleaning Started")
print("="*50)

# 1. Remove duplicates
df_cleaned = df_with_issues.drop_duplicates()
print(f"After removing duplicates: {df_cleaned.shape[0]} rows")

# 2. Handle missing values
initial_missing = df_cleaned['feedback_text'].isnull().sum()
df_cleaned = df_cleaned.dropna(subset=['feedback_text'])
print(f"After removing missing feedback_text: {df_cleaned.shape[0]} rows (removed {initial_missing} rows)")

# 3. Remove special characters and clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove excessive special characters (but keep punctuation for sentence structure)
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    # Remove multiple punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df_cleaned['feedback_text_cleaned'] = df_cleaned['feedback_text'].apply(clean_text)
print(f"Text cleaning completed")

# Save cleaned dataset
df_cleaned.to_csv('customer_feedback_cleaned.csv', index=False)

print(f"\n" + "="*50)
print("STEP 3: Preprocessing Statistics")
print("="*50)
print(f"Original dataset: {df_with_issues.shape[0]} rows")
print(f"Duplicates removed: {df_with_issues.duplicated().sum()} rows")
print(f"Missing values removed: {initial_missing} rows")
print(f"Final cleaned dataset: {df_cleaned.shape[0]} rows")
print(f"\nSample of cleaned text:")
print(df_cleaned[['feedback_text', 'feedback_text_cleaned']].head(3))

print(f"\n✓ Data preprocessing completed!")
print(f"✓ Cleaned dataset saved as 'customer_feedback_cleaned.csv'")
