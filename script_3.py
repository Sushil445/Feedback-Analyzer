
# Part 2: Create BERT Sentiment Classification Model Script

bert_model_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Customer Feedback Sentiment Classification using BERT
======================================================
This script builds and trains a BERT-based sentiment classification model
for customer feedback analysis.

Model: DistilBERT (distilbert-base-uncased)
Classes: Positive, Negative, Neutral
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class FeedbackDataset(Dataset):
    """Custom Dataset for customer feedback"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier:
    """BERT-based Sentiment Classifier for Customer Feedback"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        self.label_map = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_data(self, df, text_column='feedback_text_cleaned', label_column='sentiment'):
        """Prepare data for training"""
        
        # Encode labels
        df['label_encoded'] = df[label_column].map(self.label_map)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df[text_column].values,
            df['label_encoded'].values,
            test_size=0.3,
            random_state=42,
            stratify=df['label_encoded']
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )
        
        print(f"Train set: {len(train_texts)} samples")
        print(f"Validation set: {len(val_texts)} samples")
        print(f"Test set: {len(test_texts)} samples")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    
    def create_data_loaders(self, train_texts, val_texts, train_labels, val_labels, batch_size=16):
        """Create PyTorch data loaders"""
        
        train_dataset = FeedbackDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = FeedbackDataset(val_texts, val_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, data_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Train the model"""
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        print("\\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            val_predictions, val_labels = self.evaluate(val_loader)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model('sentiment_model_best.pt')
                print("✓ Model saved!")
        
        print("\\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
    
    def save_model(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_map': self.label_map
        }, path)
    
    def load_model(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_map = checkpoint['label_map']
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def predict(self, texts):
        """Predict sentiments for new texts"""
        self.model.eval()
        predictions = []
        
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, pred = torch.max(outputs.logits, dim=1)
                predictions.append(self.reverse_label_map[pred.item()])
        
        return predictions

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("BERT SENTIMENT CLASSIFICATION MODEL")
    print("="*60)
    
    # Load preprocessed data
    print("\\nLoading data...")
    df = pd.read_csv('customer_feedback_cleaned.csv')
    print(f"Loaded {len(df)} records")
    
    # Initialize classifier
    print("\\nInitializing BERT classifier...")
    classifier = SentimentClassifier()
    
    # Prepare data
    print("\\nPreparing data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        classifier.prepare_data(df)
    
    # Create data loaders
    print("\\nCreating data loaders...")
    train_loader, val_loader = classifier.create_data_loaders(
        train_texts, val_texts, train_labels, val_labels, batch_size=16
    )
    
    # Train model
    classifier.train(train_loader, val_loader, epochs=3)
    
    # Create test loader and evaluate
    print("\\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_dataset = FeedbackDataset(test_texts, test_labels, classifier.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    test_predictions, test_labels_array = classifier.evaluate(test_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels_array, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_array, test_predictions, average='weighted'
    )
    
    print(f"\\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    print("\\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    target_names = ['Positive', 'Neutral', 'Negative']
    print(classification_report(test_labels_array, test_predictions, target_names=target_names))
    
    # Test predictions on sample texts
    print("\\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_texts = [
        "This product is amazing! Highly recommend.",
        "Terrible experience, very disappointed.",
        "It's okay, nothing special."
    ]
    
    predictions = classifier.predict(sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"\\nText: {text}")
        print(f"Predicted Sentiment: {pred}")
    
    print("\\n" + "="*60)
    print("✓ MODEL TRAINING AND EVALUATION COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
'''

# Save the BERT model script
with open('sentiment_classification_bert.py', 'w', encoding='utf-8') as f:
    f.write(bert_model_code)

print("✓ Created 'sentiment_classification_bert.py' script")
print("\nScript includes:")
print("  - DistilBERT model for sentiment classification")
print("  - Custom PyTorch Dataset class")
print("  - Training and evaluation functions")
print("  - Metrics: Accuracy, Precision, Recall, F1-Score")
print("  - Model saving and loading")
print("  - Prediction on new texts")
print("\nThe script is ready to run with PyTorch and Transformers!")
