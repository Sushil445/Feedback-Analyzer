
# Part 5: Create Streamlit Web Application

streamlit_app_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Customer Feedback Analysis - Streamlit Web Application
=======================================================
A comprehensive web application for customer feedback analysis with:
- File upload capability
- Sentiment analysis visualization
- Text summarization
- Predictive insights dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

class FeedbackAnalyzer:
    """Main application class for feedback analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentiment_labels = ['Positive', 'Neutral', 'Negative']
    
    @st.cache_resource
    def load_sentiment_model(_self):
        """Load pre-trained sentiment model"""
        try:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=3
            )
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    
    def predict_sentiment(self, text, tokenizer, model):
        """Predict sentiment for a single text"""
        if not text or pd.isna(text):
            return 'Neutral', 0.33
        
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        return self.sentiment_labels[prediction.item()], confidence.item()
    
    def create_sentiment_distribution_chart(self, df):
        """Create sentiment distribution pie chart"""
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c']
        )])
        
        fig.update_layout(
            title='Sentiment Distribution',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_rating_distribution_chart(self, df):
        """Create rating distribution bar chart"""
        rating_counts = df['rating'].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color='#3498db',
            text=rating_counts.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Rating Distribution',
            xaxis_title='Rating',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def create_category_sentiment_chart(self, df):
        """Create category-wise sentiment heatmap"""
        category_sentiment = pd.crosstab(df['category'], df['sentiment'])
        
        fig = go.Figure(data=go.Heatmap(
            z=category_sentiment.values,
            x=category_sentiment.columns,
            y=category_sentiment.index,
            colorscale='RdYlGn',
            text=category_sentiment.values,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Category-wise Sentiment Analysis',
            height=400,
            xaxis_title='Sentiment',
            yaxis_title='Category'
        )
        
        return fig
    
    def create_time_series_chart(self, df):
        """Create time series chart of satisfaction scores"""
        df['date'] = pd.to_datetime(df['date'])
        daily_scores = df.groupby('date')['satisfaction_score'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_scores['date'],
            y=daily_scores['satisfaction_score'],
            mode='lines+markers',
            name='Average Satisfaction',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        ))
        
        # Add trend line
        z = np.polyfit(range(len(daily_scores)), daily_scores['satisfaction_score'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=daily_scores['date'],
            y=p(range(len(daily_scores))),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Satisfaction Score Over Time',
            xaxis_title='Date',
            yaxis_title='Satisfaction Score',
            height=400,
            hovermode='x unified'
        )
        
        return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Customer Feedback Analyzer</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application provides comprehensive analysis of customer feedback with:
    - **Sentiment Analysis**: Classify feedback as Positive, Negative, or Neutral
    - **Visual Analytics**: Interactive charts and insights
    - **Trend Analysis**: Track satisfaction over time
    """)
    
    # Initialize analyzer
    analyzer = FeedbackAnalyzer()
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your feedback CSV file",
        type=['csv'],
        help="Upload a CSV file with customer feedback data"
    )
    
    # Use sample data if no file uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✓ Loaded {len(df)} records")
    else:
        st.sidebar.info("📝 Using sample data")
        # Load sample data
        try:
            df = pd.read_csv('customer_feedback_cleaned.csv')
        except:
            st.error("No data available. Please upload a CSV file.")
            st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Sentiment filter
    sentiments = ['All'] + list(df['sentiment'].unique())
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment", sentiments)
    
    # Category filter
    if 'category' in df.columns:
        categories = ['All'] + list(df['category'].unique())
        selected_category = st.sidebar.selectbox("Filter by Category", categories)
    else:
        selected_category = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    if selected_category != 'All' and 'category' in df.columns:
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📈 Sentiment Analysis",
        "📝 Feedback Details",
        "🔮 Insights"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.markdown('<div class="sub-header">Key Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Feedback",
                len(filtered_df),
                delta=None
            )
        
        with col2:
            avg_rating = filtered_df['rating'].mean()
            st.metric(
                "Average Rating",
                f"{avg_rating:.2f}",
                delta=f"{avg_rating - 3:.2f}"
            )
        
        with col3:
            if 'satisfaction_score' in filtered_df.columns:
                avg_satisfaction = filtered_df['satisfaction_score'].mean()
                st.metric(
                    "Avg Satisfaction",
                    f"{avg_satisfaction:.1f}%",
                    delta=f"{avg_satisfaction - 70:.1f}%"
                )
        
        with col4:
            positive_pct = (filtered_df['sentiment'] == 'Positive').sum() / len(filtered_df) * 100
            st.metric(
                "Positive Feedback",
                f"{positive_pct:.1f}%",
                delta=f"{positive_pct - 50:.1f}%"
            )
        
        st.markdown('<div class="sub-header">Visual Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = analyzer.create_sentiment_distribution_chart(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = analyzer.create_rating_distribution_chart(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        if 'category' in filtered_df.columns:
            fig3 = analyzer.create_category_sentiment_chart(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
        
        if 'date' in filtered_df.columns:
            fig4 = analyzer.create_time_series_chart(filtered_df)
            st.plotly_chart(fig4, use_container_width=True)
    
    # Tab 2: Sentiment Analysis
    with tab2:
        st.markdown('<div class="sub-header">Sentiment Analysis</div>', unsafe_allow_html=True)
        
        # Sentiment breakdown
        col1, col2, col3 = st.columns(3)
        
        sentiment_counts = filtered_df['sentiment'].value_counts()
        
        with col1:
            positive_count = sentiment_counts.get('Positive', 0)
            positive_pct = (positive_count / len(filtered_df)) * 100
            st.metric("😊 Positive", positive_count, f"{positive_pct:.1f}%")
        
        with col2:
            neutral_count = sentiment_counts.get('Neutral', 0)
            neutral_pct = (neutral_count / len(filtered_df)) * 100
            st.metric("😐 Neutral", neutral_count, f"{neutral_pct:.1f}%")
        
        with col3:
            negative_count = sentiment_counts.get('Negative', 0)
            negative_pct = (negative_count / len(filtered_df)) * 100
            st.metric("😞 Negative", negative_count, f"{negative_pct:.1f}%")
        
        # Sentiment by category
        if 'category' in filtered_df.columns:
            st.markdown("### Sentiment by Category")
            category_sentiment = filtered_df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            st.bar_chart(category_sentiment)
    
    # Tab 3: Feedback Details
    with tab3:
        st.markdown('<div class="sub-header">Feedback Records</div>', unsafe_allow_html=True)
        
        # Display filtered data
        display_columns = ['feedback_text', 'sentiment', 'rating', 'category', 'date']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_columns].head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_feedback.csv",
            mime="text/csv"
        )
    
    # Tab 4: Insights
    with tab4:
        st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)
        
        # Top issues from negative feedback
        if 'Negative' in filtered_df['sentiment'].values:
            st.markdown("### ⚠️ Top Issues (Negative Feedback)")
            negative_feedback = filtered_df[filtered_df['sentiment'] == 'Negative']
            
            if 'category' in negative_feedback.columns:
                issue_counts = negative_feedback['category'].value_counts().head(5)
                fig_issues = go.Figure(data=[go.Bar(
                    x=issue_counts.values,
                    y=issue_counts.index,
                    orientation='h',
                    marker_color='#e74c3c'
                )])
                fig_issues.update_layout(
                    title='Top Issue Categories',
                    xaxis_title='Count',
                    yaxis_title='Category',
                    height=300
                )
                st.plotly_chart(fig_issues, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if negative_pct > 30:
            st.warning("⚠️ **High Priority**: Negative feedback exceeds 30%. Immediate action recommended.")
        elif negative_pct > 20:
            st.info("📌 **Moderate Priority**: Consider addressing recurring complaints.")
        else:
            st.success("✅ **Good Performance**: Maintain current quality standards.")
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sentiment Distribution:**")
            for sentiment, count in sentiment_counts.items():
                pct = (count / len(filtered_df)) * 100
                st.write(f"- {sentiment}: {count} ({pct:.1f}%)")
        
        with col2:
            if 'rating' in filtered_df.columns:
                st.write("**Rating Statistics:**")
                st.write(f"- Mean: {filtered_df['rating'].mean():.2f}")
                st.write(f"- Median: {filtered_df['rating'].median():.0f}")
                st.write(f"- Mode: {filtered_df['rating'].mode()[0]:.0f}")

if __name__ == "__main__":
    main()
'''

# Save the Streamlit app
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_app_code)

print("✓ Created 'streamlit_app.py' - Streamlit web application")
print("\nFeatures included:")
print("  - File upload capability")
print("  - Interactive sentiment analysis dashboard")
print("  - Multiple visualization charts")
print("  - Real-time filtering")
print("  - Feedback details table")
print("  - Insights and recommendations")
print("  - Data export functionality")
print("\nTo run: streamlit run streamlit_app.py")
