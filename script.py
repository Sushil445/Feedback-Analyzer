
# Part 1: Generate synthetic customer feedback dataset (1000+ records)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Categories for feedback
categories = ['Product Quality', 'Customer Service', 'Delivery', 'Pricing', 'User Experience', 'Features']
products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
sentiments = ['Positive', 'Negative', 'Neutral']

# Sample feedback templates
positive_feedback = [
    "Excellent product! Really satisfied with the quality and service.",
    "Amazing experience! The product exceeded my expectations.",
    "Great value for money. Highly recommend to everyone!",
    "Outstanding customer service. They resolved my issue quickly.",
    "The delivery was fast and the product is exactly as described.",
    "I love this product! It's exactly what I needed.",
    "Five stars! The quality is superb and the price is reasonable.",
    "Fantastic! Will definitely order again.",
    "Very happy with my purchase. Great product!",
    "The team was very helpful and professional."
]

negative_feedback = [
    "Very disappointed with the product quality. Not worth the price.",
    "Poor customer service. They didn't respond to my complaints.",
    "The delivery took too long and the product was damaged.",
    "Terrible experience. The product doesn't work as advertised.",
    "Overpriced and low quality. Would not recommend.",
    "The features are missing and the interface is confusing.",
    "Bad experience. The product arrived late and broken.",
    "Customer support is unhelpful and rude.",
    "Complete waste of money. Very unsatisfied.",
    "The product quality is subpar and doesn't meet expectations."
]

neutral_feedback = [
    "The product is okay, nothing special but does the job.",
    "Average experience. Some features work well, others don't.",
    "It's decent but could be better. Price is fair.",
    "The product meets basic expectations but nothing more.",
    "Neutral feelings about this. It works but not impressive.",
    "Average quality for the price. Neither good nor bad.",
    "The experience was okay. Some issues but manageable.",
    "It's fine, works as expected but nothing exceptional.",
    "Mixed feelings. Some aspects are good, others need improvement.",
    "Acceptable product but room for improvement in several areas."
]

# Generate dataset
num_records = 1200
data = []

for i in range(num_records):
    # Random date within last 2 years
    start_date = datetime.now() - timedelta(days=730)
    random_days = random.randint(0, 730)
    feedback_date = start_date + timedelta(days=random_days)
    
    # Select sentiment with weighted probability
    sentiment = random.choices(sentiments, weights=[0.5, 0.3, 0.2])[0]
    
    # Select feedback based on sentiment
    if sentiment == 'Positive':
        feedback = random.choice(positive_feedback)
        rating = random.randint(4, 5)
        satisfaction_score = random.randint(80, 100)
    elif sentiment == 'Negative':
        feedback = random.choice(negative_feedback)
        rating = random.randint(1, 2)
        satisfaction_score = random.randint(10, 40)
    else:
        feedback = random.choice(neutral_feedback)
        rating = 3
        satisfaction_score = random.randint(50, 70)
    
    # Create record
    record = {
        'feedback_id': f'FB{1000 + i}',
        'customer_id': f'CUST{random.randint(1000, 9999)}',
        'product': random.choice(products),
        'category': random.choice(categories),
        'feedback_text': feedback,
        'rating': rating,
        'sentiment': sentiment,
        'satisfaction_score': satisfaction_score,
        'date': feedback_date.strftime('%Y-%m-%d'),
        'region': random.choice(['North', 'South', 'East', 'West']),
        'customer_type': random.choice(['New', 'Returning', 'Premium'])
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('customer_feedback_raw.csv', index=False)

print(f"Generated {len(df)} customer feedback records")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())
print(f"\nRating distribution:")
print(df['rating'].value_counts())
