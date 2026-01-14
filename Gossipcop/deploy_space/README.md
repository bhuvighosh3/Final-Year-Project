---
title: Gossipcop Fake News Detection
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# Gossipcop Fake News Detection and Propagation

This project uses Graph Neural Networks (GNNs) and Social Network Analysis (SNA) to detect fake news and simulate its spread.

## Features
- **Fake News Detection**: Uses a GAT (Graph Attention Network) model.
- **Propagation Simulation**: Simulates rumor spread using the Independent Cascade (IC) model.
- **Visualizations**: Interactive 2D/3D graphs using Plotly.

## Data
The project uses the UPFD (User Preference-aware Fake News Detection) dataset. On the first run, the application will automatically download the necessary graph data.
