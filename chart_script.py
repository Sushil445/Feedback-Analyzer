import plotly.graph_objects as go
import plotly.express as px

# Create a flowchart-style diagram using Plotly
fig = go.Figure()

# Define node positions and details
nodes = {
    'Input': {'pos': (2, 9), 'text': 'Customer Feedback<br>Data Sources<br>(CSV/JSON/DB)', 'color': '#B3E5EC'},
    'Preprocess': {'pos': (2, 7.5), 'text': 'Data Preprocessing<br>& Text Processing', 'color': '#B3E5EC'},
    'Sentiment': {'pos': (0.5, 6), 'text': 'Sentiment Analysis<br>BERT/DistilBERT', 'color': '#A5D6A7'},
    'Summary': {'pos': (2, 6), 'text': 'Text Summarization<br>T5 Transformer', 'color': '#FFEB8A'},
    'Predict': {'pos': (3.5, 6), 'text': 'Predictive Insights<br>Issue Detection', 'color': '#E1BEE7'},
    'Training': {'pos': (0.5, 4.5), 'text': 'Model Training<br>Evaluation Metrics', 'color': '#A5D6A7'},
    'Extract': {'pos': (2, 4.5), 'text': 'Summary Generation<br>TF-IDF Extraction', 'color': '#FFEB8A'},
    'Forecast': {'pos': (3.5, 4.5), 'text': 'Prophet Forecasting<br>30-day Predictions', 'color': '#E1BEE7'},
    'Dashboard': {'pos': (2, 3), 'text': 'Streamlit Dashboard<br>Web Interface', 'color': '#FFCDD2'},
    'Output': {'pos': (2, 1.5), 'text': 'Final Outputs<br>Reports & Insights', 'color': '#FFF3E0'}
}

# Define connections
connections = [
    ('Input', 'Preprocess'),
    ('Preprocess', 'Sentiment'),
    ('Preprocess', 'Summary'),
    ('Preprocess', 'Predict'),
    ('Sentiment', 'Training'),
    ('Summary', 'Extract'),
    ('Predict', 'Forecast'),
    ('Training', 'Dashboard'),
    ('Extract', 'Dashboard'),
    ('Forecast', 'Dashboard'),
    ('Dashboard', 'Output')
]

# Add connections (arrows)
for start, end in connections:
    start_pos = nodes[start]['pos']
    end_pos = nodes[end]['pos']
    
    fig.add_annotation(
        x=end_pos[0], y=end_pos[1],
        ax=start_pos[0], ay=start_pos[1],
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666666'
    )

# Add nodes
for node_name, node_info in nodes.items():
    x, y = node_info['pos']
    text = node_info['text']
    color = node_info['color']
    
    fig.add_shape(
        type="rect",
        x0=x-0.4, y0=y-0.3,
        x1=x+0.4, y1=y+0.3,
        fillcolor=color,
        line=dict(color="#333333", width=2),
        layer="below"
    )
    
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=False,
        font=dict(size=10, color="black"),
        align="center"
    )

# Update layout
fig.update_layout(
    title="AI Customer Feedback Analysis System",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 10]),
    plot_bgcolor='white',
    showlegend=False,
    width=800,
    height=900
)

# Save the chart
fig.write_image("ai_feedback_system.png")
fig.write_image("ai_feedback_system.svg", format="svg")

print("AI Customer Feedback Analysis System flowchart created successfully!")