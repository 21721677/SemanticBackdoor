import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset


def load_dataset(args):
    dataset = TUDataset(
        root="datasets", name=args.dataset, use_node_attr=args.use_node_attr)
    dataset.has_node_attr = dataset.num_node_attributes > 0
    # node_features =  node_attributes + node_labels
    return dataset, dataset.num_node_attributes, dataset.num_node_labels


class GCN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_node_features, hidden_channels, num_classes, device):
        super(GCN, self).__init__()
        self.conv_in = GCNConv(num_node_features, hidden_channels)
        self.hidden_layers = []
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(
                GCNConv(hidden_channels, hidden_channels).to(device))
        self.conv_out = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv_in(x, edge_index)
        x = x.relu()
        for gcn_layer in self.hidden_layers:
            x = gcn_layer(x, edge_index)
            x = x.relu()
            # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_out(x, edge_index)

        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=-1)


def train_model(model, loader, opt, device):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        loss_all += data.y.size(0)*float(loss)
        opt.step()
    return loss_all


@torch.no_grad()
def test_model(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=-1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


@torch.no_grad()
def test_backdoor(model, loader, target, device):
    model.eval()

    ASR = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        ASR += int((pred == target).sum())
    return ASR / len(loader.dataset)


# find whether the target node exists
def has_node(graph, num_attributes, nodeLabel: int):
    sum_array = graph.x.sum(axis=0).numpy().astype(int)
    return sum_array[num_attributes+nodeLabel] > 0


# the prediction score (pred_score) is equal to the confidence of the sample on its own class
def predict_sample(model, data, device):
    model.eval()

    data = data.to(device)
    out = model(data)
    pred_score = out[0][data.y.item()].item()
    return pred_score


# modify features of the target node to zeros([0, 0, ..., 0])
def modify_features(graph, num_attributes, nodeLabel: int):
    n = graph.x.shape[1]
    new_graph = graph
    new_feature = np.zeros(n)
    for i, feature in enumerate(new_graph.x):
        if feature[num_attributes+nodeLabel] == 1:
            new_graph.x[i, :] = torch.from_numpy(new_feature)
    return new_graph
