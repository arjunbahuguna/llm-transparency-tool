import sys
import os
import torch
import networkx as nx

sys.path.insert(0, os.path.abspath('llm_transparency_tool'))
import llm_transparency_tool.routes.graph as graph_builder
from llm_transparency_tool.models.mt2_model import Mt2TransparentLlm

model = Mt2TransparentLlm('mt2/model_state_dict.pt', device='cpu')
model.run(['amar.wav'])

full_graph = graph_builder.build_full_graph(model, batch_i=0)
n_layers = model.model_info().n_layers
n_tokens = model.tokens()[0].shape[0]

# test threshold
threshold = 0.01

start_token = n_tokens - 1
print(f"Start token: {start_token}")

paths = graph_builder.build_paths_to_predictions(full_graph, n_layers, n_tokens, [start_token], threshold)
tree = paths[0]
print(f"Tree for {start_token}: Nodes: {tree.number_of_nodes()}, Edges: {tree.number_of_edges()}")

# try with a known small subgraph
test_g = nx.DiGraph()
test_g.add_edge("B", "A", weight=1.0) # B -> A
sub = nx.subgraph_view(test_g, filter_edge=lambda u,v: test_g[u][v]["weight"] > 0)
edges = list(nx.edge_dfs(sub, source="B"))
print("DFS edges test:", edges)

