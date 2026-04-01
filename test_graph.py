import sys
import os
import torch
import networkx as nx

# Add paths for imports
sys.path.insert(0, os.path.abspath('llm_transparency_tool'))
import llm_transparency_tool.routes.graph as graph_builder
from llm_transparency_tool.models.mt2_model import Mt2TransparentLlm

model = Mt2TransparentLlm('mt2/model_state_dict.pt', device='cpu')
model.run(['amar.wav'])

graph = graph_builder.build_full_graph(model, batch_i=0)
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
print(f"Sample edges: {list(graph.edges(data=True))[:5]}")
