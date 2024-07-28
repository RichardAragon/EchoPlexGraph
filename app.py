import os
import spacy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import json
import re

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Instantiate the OpenAI client
api_key = os.getenv('OPENAI_API_KEY')  # Fetch the API key from the environment variable
client = OpenAI(api_key='Your-API-Key')

# Define a function to preprocess text data
def preprocess_text(text):
    # Tokenize the text
    tokens = nlp(text)
    
    # Remove stop words and punctuation
    tokens = [token.text for token in tokens if not token.is_stop and not token.is_punct]
    
    # Lemmatize the tokens
    tokens = [token.lemma_ for token in nlp(' '.join(tokens))]
    
    return tokens

# Define a function to build the EchoPlex graph based on LLM analysis
def build_echoplex_graph_from_llm(analysis):
    # Create an empty graph
    G = nx.Graph()
    
    try:
        # Extract the JSON part of the analysis
        json_data = re.search(r"\[.*\]", analysis, re.DOTALL).group(0)
        concepts = json.loads(json_data)
        
        # Add nodes for each concept with their weights
        for concept in concepts:
            G.add_node(concept['word'], weight=concept['weight'])
        
        # Add edges between words based on co-occurrence
        for concept in concepts:
            for connection in concept['connections']:
                G.add_edge(concept['word'], connection['word'], weight=connection['weight'])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response content: {analysis}")
    except AttributeError as e:
        print(f"Error extracting JSON: {e}")
        print(f"Response content: {analysis}")
    
    return G

# Define a function to identify clusters and hubs in the graph
def identify_clusters_and_hubs(G):
    # Use community detection to identify clusters
    clusters = list(nx.algorithms.community.greedy_modularity_communities(G))
    
    # Identify hubs using centrality measures
    hubs = sorted(G.nodes, key=lambda node: nx.degree_centrality(G)[node], reverse=True)
    
    return clusters, hubs

# Define a function to call the OpenAI API for pattern analysis
def analyze_patterns_with_llm(texts):
    prompt = f"Analyze the following texts for patterns, providing words, their weights, and connections. Format the response as a JSON array with each word having 'word', 'weight', and 'connections' (each connection having 'word' and 'weight'):\n\n{json.dumps(texts)}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message.content.strip()
    return analysis

# Define a function to visualize the EchoPlex graph, clusters, and hubs
def visualize_graph(G, clusters, hubs, analysis):
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw nodes with sizes based on their weights
    node_sizes = [G.nodes[node]['weight'] * 1000 for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
    
    # Draw edges with widths based on their weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    # Highlight clusters
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for i, cluster in enumerate(clusters):
        nx.draw_networkx_nodes(G, pos, nodelist=list(cluster), node_size=700, node_color=colors[i % len(colors)])
    
    # Highlight hubs
    nx.draw_networkx_nodes(G, pos, nodelist=hubs[:5], node_size=1000, node_color='orange')
    
    plt.title("EchoPlex Graph with Clusters and Hubs")
    plt.show()
    
    # Print the LLM analysis
    print("\nPattern Analysis from LLM:\n", analysis)

# Test the EchoPlex implementation
texts = [
    "This is a sample text about machine learning and natural language processing.",
    "Machine learning is a field of study that focuses on the use of algorithms and statistical models to enable machines to perform a specific task.",
    "Natural language processing is a subfield of machine learning that deals with the interaction between computers and humans in natural language."
]

# Analyze patterns using LLM
analysis = analyze_patterns_with_llm(texts)
print(f"LLM Analysis Response: {analysis}")

# Build the EchoPlex graph from LLM analysis
G = build_echoplex_graph_from_llm(analysis)
clusters, hubs = identify_clusters_and_hubs(G)

# Visualize the graph and print analysis
visualize_graph(G, clusters, hubs, analysis)
