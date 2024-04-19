import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
import torch_geometric.transforms as T
from .utils import *
from functools import partial
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name in ['PATTERN', 'CLUSTER', 'CSL']:
        tf_list = []
    else:
        raise ValueError(f"Loading dataset '{name}' from "
                         f"GNNBenchmarkDataset is not supported.")

    if name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        dataset = join_dataset_splits(
            [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
            for split in ['train', 'val', 'test']]
        )
        pre_transform_in_memory(dataset, T.Compose(tf_list))
    elif name == 'CSL':
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name)

    return dataset


def weighted_cross_entropy(pred, true):
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                  weight=weight[true])
        return loss


def accuracy_sbm(scores, targets):
    S = targets.cpu().numpy()
    C = scores.cpu().numpy()
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


class GNNBenchmarkUtil:
    @staticmethod
    def add_args(parser):
        parser.set_defaults(batch_size=16)
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(weight_decay=0.0)
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(gnn_emb_dim=64)

    @staticmethod
    def loss_fn(task_type):
        def calc_loss(pred, batch, is_inductive_node, m=1.0):
            if is_inductive_node:
                loss = weighted_cross_entropy(pred, batch.y)
            else:
                loss = F.cross_entropy(pred, batch.y)
            return loss

        return calc_loss

    @staticmethod
    @torch.no_grad()
    def eval(model, device, loader, evaluator, is_inductive_node):
        model.eval()
        if is_inductive_node:
            true_list = []
            pred_list = []
            for step, batch in enumerate(loader):
                batch = batch.to(device)

                pred = model(batch)
                true_list.append(batch.y)
                pred_list.append(pred)
            true = torch.cat(true_list)
            pred_score = torch.cat(pred_list)
            pred_int = pred_score.max(dim=1)[1]
            acc = accuracy_sbm(pred_int, true)

        else:
            correct = 0
            for step, batch in enumerate(loader):
                batch = batch.to(device)

                pred = model(batch)
                pred = pred.max(dim=1)[1]
                correct += pred.eq(batch.y).sum().item()
            acc = correct / len(loader.dataset)
        return {"acc": acc}

    @staticmethod
    def preprocess(args):
        dataset = preformat_GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), args.dataset)

        num_tasks = dataset.num_classes
        num_node_features = dataset.num_node_features
        num_edge_features = dataset.num_edge_features

        training_set, validation_set, test_set = [Subset(dataset, split_idx) for split_idx in dataset.split_idxs]

        class Dataset(dict):
            pass

        dataset = Dataset({"train": training_set, "valid": validation_set, "test": test_set})
        dataset.eval_metric = "acc"
        dataset.task_type = "classification"
        dataset.get_idx_split = lambda: {"train": "train", "valid": "valid", "test": "test"}

        node_encoder_cls = lambda: nn.Linear(num_node_features, args.gnn_emb_dim)
        if args.dataset in ['MNIST', 'CIFAR10']:
            edge_encoder_cls = lambda emb_dim: nn.Linear(num_edge_features, emb_dim)
        elif args.dataset in ['PATTERN', 'CLUSTER']:
            def edge_encoder_cls(_):
                def zero(_):
                    return 0

                return zero

        return dataset, num_tasks, node_encoder_cls, edge_encoder_cls, None