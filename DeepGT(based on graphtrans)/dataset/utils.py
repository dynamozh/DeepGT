from collections import Counter

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


class ASTNodeEncoder(torch.nn.Module):
    """
    Input:
        x: default node feature. the first and second column represents node type and node attributes.
        depth: The depth of the node in the AST.

    Output:
        emb_dim-dimensional vector

    """

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)


def get_vocab_mapping(seq_list, num_vocab):
    """
    Input:
        seq_list: a list of sequences
        num_vocab: vocabulary size
    Output:
        vocab2idx:
            A dictionary that maps vocabulary into integer index.
            Additioanlly, we also index '__UNK__' and '__EOS__'
            '__UNK__' : out-of-vocabulary term
            '__EOS__' : end-of-sentence

        idx2vocab:
            A list that maps idx to actual vocabulary.

    """

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind="stable")[:num_vocab]

    logger.info("Coverage of top {} vocabulary: {:.4f}", num_vocab, float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # logger.info(topvocab)
    # logger.info([vocab_list[v] for v in topvocab[:10]])
    # logger.info([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx["__UNK__"] = num_vocab
    idx2vocab.append("__UNK__")

    vocab2idx["__EOS__"] = num_vocab + 1
    idx2vocab.append("__EOS__")

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert idx == vocab2idx[vocab]

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert vocab2idx["__EOS__"] == len(idx2vocab) - 1

    return vocab2idx, idx2vocab


def augment_edge(data):
    """
    Input:
        data: PyG data object
    Output:
        data (edges are augmented in the following ways):
            data.edge_index: Added next-token edge. The inverse edges were also added.
            data.edge_attr (torch.Long):
                data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    """

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim=1
    )

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(
        data.node_is_attributed.view(
            -1,
        )
        == 1
    )[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim=0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim=1)

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
    data.edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse], dim=0)

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    """
    Input:
        data: PyG graph object
        output: add y_arr to data
    """

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    """
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    """

    augmented_seq = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"] for w in augmented_seq]], dtype=torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    """
    Input: torch 1d array: y_arr
    Output: a sequence of words.
    """

    eos_idx_list = (arr == len(idx2vocab) - 1).nonzero()  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0].__indices__ = None
    datasets[0].__data_list__ = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset.__indices__ = None
    dataset.__data_list__ = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def test():
    seq_list = [["a", "b"], ["a", "b", "c", "df", "f", "2edea", "a"], ["eraea", "a", "c"], ["d"], ["4rq4f", "f", "a", "a", "g"]]
    vocab2idx, idx2vocab = get_vocab_mapping(seq_list, 4)
    logger.debug(vocab2idx)
    logger.debug(idx2vocab)
    assert len(vocab2idx) == len(idx2vocab)

    for vocab, idx in vocab2idx.items():
        assert idx2vocab[idx] == vocab

    for seq in seq_list:
        logger.debug(seq)
        arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len=4)[0]
        # Test the effect of predicting __EOS__
        # arr[2] = vocab2idx['__EOS__']
        logger.debug(arr)
        seq_dec = decode_arr_to_seq(arr, idx2vocab)

        logger.debug(arr)
        logger.debug(seq_dec)


if __name__ == "__main__":
    test()
