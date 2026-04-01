import sys
import os
import networkx as nx

sys.path.insert(0, os.path.abspath('llm_transparency_tool'))
import llm_transparency_tool.routes.graph as graph_builder
from llm_transparency_tool.models.mt2_model import Mt2TransparentLlm
import llm_transparency_tool.routes.contributions as contributions

model = Mt2TransparentLlm('mt2/model_state_dict.pt', device='cpu')
model.run(['amar.wav'])

B0 = 0
tokens = model.tokens()[B0]
token_strings = model.tokens_to_strings(tokens)
print("Tokens:", token_strings[:5], "...", token_strings[-5:])

full_graph = graph_builder.build_full_graph(model, batch_i=B0, renormalizing_threshold=0.0)
n_layers = model.model_info().n_layers
n_tokens = tokens.shape[0]

# Check edges for token 0 vs token 127
paths = graph_builder.build_paths_to_predictions(full_graph, n_layers, n_tokens, [0, 1, 127], 0.04) # 0.04 is the default threshold in UI
print(f"Edges for token 0: {paths[0].number_of_edges()}")
print(f"Edges for token 1: {paths[1].number_of_edges()}")
print(f"Edges for token 127: {paths[2].number_of_edges()}")

