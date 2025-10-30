import plotly.graph_objects as go

# Data for sentiment distribution
sentiment_data = {
    "labels": ["Positive", "Negative", "Neutral"],
    "values": [615, 343, 242]
}

# Calculate percentages
total = sum(sentiment_data["values"])
percentages = [round((value/total)*100, 1) for value in sentiment_data["values"]]

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=sentiment_data["labels"],
    values=sentiment_data["values"],
    marker_colors=['#2E8B57', '#DB4545', '#D2BA4C'],  # Green, Red, Yellow from brand colors
    textinfo='label+percent',
    textposition='inside'
)])

# Update layout
fig.update_layout(
    title="Dataset Sentiment Distribution",
    showlegend=True,
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Save as both PNG and SVG
fig.write_image("sentiment_distribution.png")
fig.write_image("sentiment_distribution.svg", format="svg")

print("Pie chart saved successfully!")
print(f"Total samples: {total}")
print(f"Positive: {sentiment_data['values'][0]} ({percentages[0]}%)")
print(f"Negative: {sentiment_data['values'][1]} ({percentages[1]}%)")
print(f"Neutral: {sentiment_data['values'][2]} ({percentages[2]}%)")