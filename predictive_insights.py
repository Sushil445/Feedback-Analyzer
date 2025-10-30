#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predictive Insight Generation for Customer Feedback
====================================================
This script analyzes customer feedback to:
1. Identify recurring issues and complaints
2. Predict customer satisfaction score trends using Prophet
3. Generate actionable insights

Forecasting Model: Facebook Prophet
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InsightGenerator:
    """Generate predictive insights from customer feedback"""

    def __init__(self):
        self.prophet_model = None

    def identify_recurring_issues(self, df, text_column='feedback_processed', sentiment_column='sentiment'):
        """
        Identify recurring issues and complaints from negative feedback

        Parameters:
        -----------
        df : DataFrame
            Customer feedback dataframe
        text_column : str
            Column containing processed feedback text
        sentiment_column : str
            Column containing sentiment labels
        """
        print("\n" + "="*70)
        print("IDENTIFYING RECURRING ISSUES")
        print("="*70)

        # Filter negative feedback
        negative_feedback = df[df[sentiment_column] == 'Negative']

        print(f"\nAnalyzing {len(negative_feedback)} negative feedback records...")

        # Extract common phrases using n-grams
        if text_column in negative_feedback.columns:
            vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=20, stop_words='english')

            try:
                X = vectorizer.fit_transform(negative_feedback[text_column].fillna(''))
                feature_names = vectorizer.get_feature_names_out()
                frequencies = X.sum(axis=0).A1

                # Create issue dataframe
                issues_df = pd.DataFrame({
                    'issue': feature_names,
                    'frequency': frequencies
                }).sort_values('frequency', ascending=False)

                print("\nTop 10 Recurring Issues:")
                print("-" * 70)
                for idx, row in issues_df.head(10).iterrows():
                    print(f"  {idx+1}. {row['issue']} (mentioned {int(row['frequency'])} times)")

                # Category-wise issues
                print("\nIssues by Category:")
                print("-" * 70)
                category_issues = negative_feedback.groupby('category').size().sort_values(ascending=False)
                for category, count in category_issues.items():
                    pct = (count / len(negative_feedback)) * 100
                    print(f"  {category}: {count} complaints ({pct:.1f}%)")

                return issues_df

            except Exception as e:
                print(f"Error extracting issues: {e}")
                return pd.DataFrame()
        else:
            print(f"Column '{text_column}' not found in dataframe")
            return pd.DataFrame()

    def prepare_time_series_data(self, df, date_column='date', score_column='satisfaction_score'):
        """Prepare time series data for Prophet forecasting"""

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Aggregate scores by date
        daily_scores = df.groupby(date_column)[score_column].mean().reset_index()
        daily_scores.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' columns

        return daily_scores

    def forecast_satisfaction(self, df, date_column='date', score_column='satisfaction_score', periods=30):
        """
        Forecast customer satisfaction scores using Prophet

        Parameters:
        -----------
        df : DataFrame
            Customer feedback dataframe
        date_column : str
            Column containing dates
        score_column : str
            Column containing satisfaction scores
        periods : int
            Number of days to forecast
        """
        print("\n" + "="*70)
        print("FORECASTING CUSTOMER SATISFACTION TRENDS")
        print("="*70)

        # Prepare data
        print("\nPreparing time series data...")
        ts_data = self.prepare_time_series_data(df, date_column, score_column)
        print(f"Data points: {len(ts_data)}")
        print(f"Date range: {ts_data['ds'].min()} to {ts_data['ds'].max()}")
        print(f"Average satisfaction score: {ts_data['y'].mean():.2f}")

        # Initialize and train Prophet model
        print("\nTraining Prophet forecasting model...")
        self.prophet_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )

        self.prophet_model.fit(ts_data)

        # Make future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods)

        # Forecast
        print(f"Generating {periods}-day forecast...")
        forecast = self.prophet_model.predict(future)

        # Display forecast results
        forecast_future = forecast[forecast['ds'] > ts_data['ds'].max()]

        print("\n" + "="*70)
        print("FORECAST RESULTS (Next 30 Days)")
        print("="*70)

        print(f"\nForecasted Average Satisfaction Score: {forecast_future['yhat'].mean():.2f}")
        print(f"Current Average Satisfaction Score: {ts_data['y'].mean():.2f}")

        trend = "INCREASING" if forecast_future['yhat'].mean() > ts_data['y'].mean() else "DECREASING"
        change = abs(forecast_future['yhat'].mean() - ts_data['y'].mean())

        print(f"\nTrend: {trend} (±{change:.2f} points)")

        # Confidence intervals
        print(f"\nConfidence Interval:")
        print(f"  Lower bound: {forecast_future['yhat_lower'].mean():.2f}")
        print(f"  Upper bound: {forecast_future['yhat_upper'].mean():.2f}")

        return forecast, ts_data

    def generate_visualizations(self, forecast, ts_data):
        """Generate forecast visualizations"""

        print("\nGenerating visualizations...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Customer Satisfaction Analysis and Forecast', fontsize=16, fontweight='bold')

        # Plot 1: Historical and Forecasted Satisfaction
        ax1 = axes[0, 0]
        ax1.plot(ts_data['ds'], ts_data['y'], label='Historical', color='blue', linewidth=2)
        forecast_future = forecast[forecast['ds'] > ts_data['ds'].max()]
        ax1.plot(forecast_future['ds'], forecast_future['yhat'], label='Forecast', color='red', linewidth=2)
        ax1.fill_between(
            forecast_future['ds'],
            forecast_future['yhat_lower'],
            forecast_future['yhat_upper'],
            alpha=0.3,
            color='red',
            label='Confidence Interval'
        )
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Satisfaction Score')
        ax1.set_title('Satisfaction Score Forecast')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trend Component
        ax2 = axes[0, 1]
        ax2.plot(forecast['ds'], forecast['trend'], color='green', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend Component')
        ax2.set_title('Overall Trend')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Weekly Seasonality
        ax3 = axes[1, 0]
        if 'weekly' in forecast.columns:
            weekly_data = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax3.bar(days, weekly_data.values, color='purple', alpha=0.7)
            ax3.set_xlabel('Day of Week')
            ax3.set_ylabel('Weekly Effect')
            ax3.set_title('Weekly Seasonality Pattern')
            ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Forecast Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        current_avg = ts_data['y'].mean()
        forecast_avg = forecast_future['yhat'].mean()
        change_pct = ((forecast_avg - current_avg) / current_avg) * 100

        summary_text = f"""
        FORECAST SUMMARY
        {'='*40}

        Current Avg Score:     {current_avg:.2f}
        Forecasted Avg Score:  {forecast_avg:.2f}
        Expected Change:       {change_pct:+.2f}%

        Forecast Period:       30 days
        Confidence:            95%

        Lower Bound:           {forecast_future['yhat_lower'].mean():.2f}
        Upper Bound:           {forecast_future['yhat_upper'].mean():.2f}

        Trend Direction:       {'↑ Improving' if change_pct > 0 else '↓ Declining'}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig('satisfaction_forecast.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: 'satisfaction_forecast.png'")

        return fig

    def generate_insights_report(self, df, issues_df, forecast, ts_data):
        """Generate comprehensive insights report"""

        print("\n" + "="*70)
        print("GENERATING INSIGHTS REPORT")
        print("="*70)

        report = []

        # Overall Statistics
        report.append("="*70)
        report.append("CUSTOMER FEEDBACK ANALYSIS - INSIGHTS REPORT")
        report.append("="*70)
        report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {df['date'].min()} to {df['date'].max()}")
        report.append(f"Total Feedback Records: {len(df)}")

        # Sentiment Distribution
        report.append("\n" + "-"*70)
        report.append("SENTIMENT DISTRIBUTION")
        report.append("-"*70)
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            report.append(f"  {sentiment}: {count} ({pct:.1f}%)")

        # Top Issues
        report.append("\n" + "-"*70)
        report.append("TOP RECURRING ISSUES")
        report.append("-"*70)
        if not issues_df.empty:
            for idx, row in issues_df.head(10).iterrows():
                report.append(f"  {idx+1}. {row['issue']} (frequency: {int(row['frequency'])})")

        # Forecast Insights
        report.append("\n" + "-"*70)
        report.append("SATISFACTION FORECAST (Next 30 Days)")
        report.append("-"*70)

        current_avg = ts_data['y'].mean()
        forecast_future = forecast[forecast['ds'] > ts_data['ds'].max()]
        forecast_avg = forecast_future['yhat'].mean()
        change = forecast_avg - current_avg
        change_pct = (change / current_avg) * 100

        report.append(f"  Current Average: {current_avg:.2f}")
        report.append(f"  Forecasted Average: {forecast_avg:.2f}")
        report.append(f"  Expected Change: {change:+.2f} ({change_pct:+.2f}%)")
        report.append(f"  Trend: {'Improving' if change > 0 else 'Declining'}")

        # Recommendations
        report.append("\n" + "-"*70)
        report.append("KEY RECOMMENDATIONS")
        report.append("-"*70)

        negative_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100

        if negative_pct > 30:
            report.append("  ⚠ HIGH PRIORITY: Negative feedback exceeds 30%")
            report.append("    → Immediate action required to address customer complaints")

        if change < 0:
            report.append("  ⚠ WARNING: Satisfaction scores are forecasted to decline")
            report.append("    → Implement retention strategies and address top issues")
        else:
            report.append("  ✓ POSITIVE: Satisfaction scores are forecasted to improve")
            report.append("    → Continue current strategies and monitor progress")

        if not issues_df.empty:
            top_issue = issues_df.iloc[0]['issue']
            report.append(f"  → Focus on addressing: '{top_issue}' (most frequent issue)")

        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)

        # Save report
        report_text = '\n'.join(report)
        with open('AI_insights_report.txt', 'w') as f:
            f.write(report_text)

        print("✓ Saved report: 'AI_insights_report.txt'")

        return report_text

def main():
    """Main pipeline for predictive insights"""

    print("="*70)
    print("PREDICTIVE INSIGHT GENERATION")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('customer_feedback_cleaned.csv')
    print(f"Loaded {len(df)} records")

    # Initialize insight generator
    generator = InsightGenerator()

    # Identify recurring issues
    issues_df = generator.identify_recurring_issues(df)

    # Forecast satisfaction trends
    forecast, ts_data = generator.forecast_satisfaction(df)

    # Generate visualizations
    generator.generate_visualizations(forecast, ts_data)

    # Generate comprehensive report
    report = generator.generate_insights_report(df, issues_df, forecast, ts_data)

    print("\n" + report)

    print("\n" + "="*70)
    print("✓ PREDICTIVE INSIGHTS GENERATION COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - AI_insights_report.txt")
    print("  - satisfaction_forecast.png")

if __name__ == "__main__":
    main()
