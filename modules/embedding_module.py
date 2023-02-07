import torch
from torch import nn
import numpy as np
import math


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, V=None, R=None):
    pass


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device):
    super(TimeEmbedding, self).__init__(node_features, edge_features,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, V=None, R=None):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device)

    self.use_memory = use_memory
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True, V=None, R=None, roots=None, targets=None, node_level=False):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """


    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)
    if roots is None:
        roots = source_nodes_torch

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    # neighbors_torch together with source_nodes_torch is what we want
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      roots = roots.expand(n_neighbors, roots.shape[0]).T.flatten()
      targets = targets.expand(n_neighbors, targets.shape[0]).T.flatten()

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors, V=V, R=R, roots=roots, targets=targets,
                                                   node_level=node_level)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask,
                                        edge_deltas_torch, V=V, R=R,
                                        src_idx=source_nodes_torch,
                                        neigh_idx=neighbors_torch,
                                        roots=roots, targets=targets, node_level=node_level)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask, timestamps, V=None, R=None,src_idx=None, neigh_idx=None,
                roots=None, targets=None, node_level=False):
    return None


class PINT(GraphEmbedding):
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               use_memory=True, beta=0.1, r_dim=4):
    super(PINT, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            use_memory=use_memory)

    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension +
                                                         n_edge_features + 2 * r_dim, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_11 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension, embedding_dimension)
                                          for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
          [torch.nn.Linear(embedding_dimension + n_node_features + 2 * r_dim,
                           embedding_dimension) for _ in range(n_layers)])

    self.linear_22 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension,
                       embedding_dimension) for _ in range(n_layers)])
    self.alpha = 2
    self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=False)

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask, timestamps, V=None, R=None, src_idx=None, neigh_idx=None,
                roots=None, targets=None, node_level=False):
    neighbors_features = torch.cat([neighbor_embeddings, edge_features],
                                   dim=2)
    mask = ~mask
    timestamps = timestamps.unsqueeze(-1)
    number_neighbors = torch.sum(mask, dim=1).unsqueeze(-1).unsqueeze(-1)

    R = R / (R.sum(dim=2, keepdim=True) + 1e-04)

    neighbor_embeddings = self.linear_1[n_layer - 1](torch.cat(
      [neighbors_features,
       R[neigh_idx.flatten(), roots, :].view((neigh_idx.shape[0], neigh_idx.shape[1], R.shape[2])),
       R[neigh_idx.flatten(), targets, :].view((neigh_idx.shape[0], neigh_idx.shape[1], R.shape[2]))],
      dim=2))

    neighbor_embeddings = torch.relu(neighbor_embeddings)
    neighbor_embeddings = self.linear_11[n_layer - 1](neighbor_embeddings)

    neighbors_sum = neighbor_embeddings * (self.alpha ** (
      -torch.relu(self.beta * (timestamps)) / torch.sqrt(number_neighbors + 1e-4)))

    neighbors_sum = torch.sum(neighbors_sum * mask.unsqueeze(-1), dim=1)

    source_embedding = torch.cat([neighbors_sum, source_node_features, R[src_idx, roots[0::neigh_idx.shape[1]], :],
                                    R[src_idx, targets[0::neigh_idx.shape[1]], :]], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)
    source_embedding = torch.relu(source_embedding)
    source_embedding = self.linear_22[n_layer - 1](source_embedding)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_neighbors=None,
                         use_memory=True, beta=0.1, r_dim=4):
  if module_type == "PINT":
    return PINT(node_features=node_features,
                              edge_features=edge_features,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              use_memory=use_memory, beta=beta, r_dim=r_dim)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


