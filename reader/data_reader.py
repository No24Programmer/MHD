import collections
import json
import random
import torch

from operator import itemgetter
from tqdm import tqdm

from reader.entity.GNNDataset import GNNDataset
from reader.entity.hfact_example import HFactExample
from reader.entity.hfact_feature_bert import HFactFeatureBERT
from reader.entity.hfact_feature_gnn import HFactFeatureGNN
from utils.train_args import get_max_node


def load_data(args, dataset_file_path, vocabulary, vocabulary_dict, device, tokenizer=None):
    """
    load data from file

    :param args:
    :param dataset_file_path: file path
    :param vocabulary:
    :param vocabulary_dict:
    :param device:
    :param tokenizer:
    :return:
    """
    examples, max_node_num = read_examples(dataset_file_path)
    if args.model_type in ["GNN"]:
        max_hypergraph_node_num = get_max_node(args.dataset_name, args.sample_k)
        gnn_dataset = convert_examples_for_GNN(examples, vocabulary, max_node_num, max_hypergraph_node_num, device, args.sample_k)
        return gnn_dataset


def read_examples(input_file):
    """
    Read an n-ary json file into a list of HFactExample.
    """
    examples = []
    max_node_num = 0
    max_arity = 0
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            assert "N" in obj.keys() \
                   and "relation" in obj.keys() \
                   and "subject" in obj.keys() \
                   and "object" in obj.keys(), \
                "There are 4 mandatory fields: N, relation, subject, and object."
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            node_num = 3

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                # store attributes in alphabetical order
                for attribute in sorted(obj.keys()):
                    if attribute == "N" \
                            or attribute == "relation" \
                            or attribute == "subject" \
                            or attribute == "object":
                        continue
                    # store corresponding values in alphabetical order
                    auxiliary_info[attribute] = sorted(obj[attribute])

                    node_num = node_num + 1 + len(obj[attribute])
            max_node_num = node_num if node_num > max_node_num else max_node_num
            max_arity = arity if arity > max_arity else max_arity

            example = HFactExample(
                arity=arity,
                node_num=node_num,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info)
            examples.append(example)

    return examples, max_node_num


def get_original_inputs(example, max_node_num=0):
    """

    :param example:
    :param max_node_num:
    :return: orig_input_tokens, orig_input_mask, orig_type_label
    """
    hrt = [example.head, example.relation, example.tail]
    hrt_mask = [1, 1, 1]

    orig_input_tokens = []
    orig_input_mask = []
    orig_type_label = [1, -1, 1]  # relation/key = -1; entity/value = 1; other = 0
    orig_element = [0, 1, 2]  # 0-object, 1-relation, 2-subject, 3-attribute, 4-value, -1 other

    orig_input_tokens.extend(hrt)
    orig_input_mask.extend(hrt_mask)

    node_id = 3
    if example.auxiliary_info is not None:
        for index, attribute in enumerate(example.auxiliary_info.keys()):
            orig_input_tokens.append(attribute)
            orig_input_mask.append(1)
            orig_type_label.append(-1)
            orig_element.append(3)
            node_id += 1
            for value in example.auxiliary_info[attribute]:
                orig_input_tokens.append(value)
                orig_input_mask.append(1)
                orig_type_label.append(1)
                orig_element.append(4)
                node_id += 1
    assert node_id == example.node_num
    if max_node_num > 0:
        while len(orig_input_tokens) < max_node_num:
            orig_input_tokens.append("[PAD]")
            orig_input_mask.append(0)
            orig_type_label.append(0)
            orig_element.append(-1)
        assert len(orig_input_tokens) == max_node_num

    return orig_input_tokens, orig_input_mask, orig_type_label, orig_element


def generate_ground_truth(ground_truth_path, vocabulary, max_node_num, vocabulary_dict=None):
    """
    Generate ground truth for filtered evaluation.
    """
    # max_aux = max_arity - 2
    # assert max_seq_length == 2 * max_aux + 3, \
    #     "Each input sequence contains relation, head, tail, " \
    #     "and max_aux attribute-value pairs."

    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    all_examples, _, = read_examples(ground_truth_path)
    for (example_id, example) in enumerate(all_examples):
        # get padded input tokens and ids
        rht = [example.head, example.relation, example.tail]
        input_tokens = rht
        # aux_attributes = []
        # aux_values = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                # if dataset != "jf17k":
                input_tokens.append(attribute)
                for value in example.auxiliary_info[attribute]:
                    input_tokens.append(value)
        # if example.auxiliary_info is not None:
        #     for attribute in example.auxiliary_info.keys():
        #         for value in example.auxiliary_info[attribute]:
        #             aux_attributes.append(attribute)
        #             aux_values.append(value)

        while len(input_tokens) < max_node_num:
            input_tokens.append("[PAD]")

        # input_tokens = rht + aux_attributes + aux_values
        if vocabulary_dict is None:
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
        else:
            trans_tokens = itemgetter(*input_tokens)(vocabulary_dict)
            input_ids = vocabulary.convert_tokens_to_ids(trans_tokens)

        assert len(input_tokens) == max_node_num
        assert len(input_ids) == max_node_num

        # get target answer for each pos and the corresponding key
        for pos in range(max_node_num):
            if input_ids[pos] == 0:
                continue
            key = " ".join([
                str(input_ids[x]) for x in range(max_node_num) if x != pos
            ])
            gt_dict[pos][key].append(input_ids[pos])

    return gt_dict


def generate_token_id_dict(examples, vocabulary):
    token_id_dict = collections.defaultdict(list)  # {token_id : [example_id]}
    vocab_id_dict = collections.defaultdict(list)
    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        orig_input_tokens, orig_input_mask, orig_type_label, orig_element = get_original_inputs(example)
        example_tokens = [token for token in orig_input_tokens if token != "[PAD]"]
        tokens_ids = vocabulary.convert_tokens_to_ids(example_tokens)
        vocab_id_dict[example_id] = tokens_ids
        for i in tokens_ids:
            token_id_dict[i].append(example_id)
    return token_id_dict, vocab_id_dict


def convert_examples_for_GNN(examples, vocabulary, max_node_num, max_hypergraph_node_num, device, k=5):
    features = []
    feature_id = 0
    token_id_dict, vocab_id_dict = generate_token_id_dict(examples, vocabulary)
    max_length = 0
    for (example_id, example) in tqdm(enumerate(examples), total=len(examples)):
        '''Stap 1: get original input tokens and input mask '''
        orig_input_tokens, orig_input_mask, orig_type_label, orig_element = get_original_inputs(example, max_node_num)

        '''Stap 2: get example neighbor'''
        h_neighbor_set = set()
        r_neighbor_set = set()
        t_neighbor_set = set()
        av_neighbor_set = set()

        for index, token in enumerate(orig_input_tokens):
            if token == "[PAD]":
                continue
            if index == 0:  # head
                h_neighbor_set.update(set(token_id_dict[vocabulary.convert_token_to_id(token)]))
            elif index == 1:    # relation
                r = set(token_id_dict[vocabulary.convert_token_to_id(token)])
                h = set(token_id_dict[vocabulary.convert_token_to_id(orig_input_tokens[0])])
                t = set(token_id_dict[vocabulary.convert_token_to_id(orig_input_tokens[2])])
                r_neighbor_set.update((r & h) | (r & t))
                # r_neighbor_set.update((r & t))
                # r_neighbor_set.update(set(token_id_dict[vocabulary.convert_token_to_id(token)]))
            elif index == 2:    # t
                t_neighbor_set.update(set(token_id_dict[vocabulary.convert_token_to_id(token)]))
            elif index > 2 and orig_type_label[index] == -1:
                a_neighbor_set = set(token_id_dict[vocabulary.convert_token_to_id(token)])
                i = index + 1
                v_neighbor_set = set()
                while i < len(orig_type_label) and orig_type_label[i] == 1:
                    v_neighbor_set.update(set(token_id_dict[vocabulary.convert_token_to_id(orig_input_tokens[i])]))
                    i = i + 1
                av_neighbor_set.update(a_neighbor_set & v_neighbor_set)

        '''Stap:3 generate a feature by masking each of the tokens  '''
        for mask_position in range(len(orig_input_tokens)):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)

            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = orig_type_label[mask_position]
            mask_element = orig_element[mask_position]

            # use [PAD] padding
            while len(input_ids) < max_node_num:
                input_ids.append(0)
            orig_input_ids = input_ids.copy()
            # while len(orig_input_ids) < max_node_num:
            #     orig_input_ids.append(0)

            ''' Step:4 generate hypergraph feature '''
            # sample
            h_neighbor_list = get_example_neighbor_list(h_neighbor_set.copy(), k, example_id, vocab_id_dict)
            r_neighbor_list = get_example_neighbor_list(r_neighbor_set.copy(), k, example_id, vocab_id_dict)
            t_neighbor_list = get_example_neighbor_list(t_neighbor_set.copy(), k, example_id, vocab_id_dict)
            av_neighbor_list = get_example_neighbor_list(av_neighbor_set.copy(), k, example_id, vocab_id_dict) if len(av_neighbor_set) != 0 else av_neighbor_set.copy()

            h_input_ids, h_incidence_matrix, h_input_mask = generate_hypergraph_feature(
                input_ids.copy(), h_neighbor_list, vocab_id_dict, max_hypergraph_node_num,
                mask_label, True if mask_element == 0 else False, k)
            r_input_ids, r_incidence_matrix, r_input_mask = generate_hypergraph_feature(
                input_ids.copy(), r_neighbor_list, vocab_id_dict, max_hypergraph_node_num,
                mask_label, True if mask_element == 1 else False, k)
            t_input_ids, t_incidence_matrix, t_input_mask = generate_hypergraph_feature(
                input_ids.copy(), t_neighbor_list, vocab_id_dict, max_hypergraph_node_num,
                mask_label, True if mask_element == 2 else False, k)
            av_input_ids, av_incidence_matrix, av_input_mask = generate_hypergraph_feature(
                input_ids.copy(), av_neighbor_list, vocab_id_dict, max_hypergraph_node_num,
                mask_label, True if mask_element in [3, 4] else False, k)
            nei_input_ids = list([h_input_ids, r_input_ids, t_input_ids, av_input_ids])
            nei_incidence_matrix = torch.stack([h_incidence_matrix, r_incidence_matrix, t_incidence_matrix, av_incidence_matrix])
            nei_input_mask = list([h_input_mask, r_input_mask, t_input_mask, av_input_mask])

            max_length = len(input_ids) if max_length < len(input_ids) else max_length

            feature = HFactFeatureGNN(
                feature_id=feature_id,
                example_id=example_id,
                input_ids=orig_input_ids,
                input_mask=orig_input_mask,
                input_label=orig_type_label,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                h_input_ids=nei_input_ids,
                h_input_mask=nei_input_mask,
                h_incidence_matrix=nei_incidence_matrix,
                arity=example.arity,
            )
            features.append(feature)
            feature_id += 1
    print(max_length)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features]).to(device)
    input_label = torch.tensor([f.input_label for f in features]).to(device)
    mask_position = torch.tensor([f.mask_position for f in features]).to(device)
    mask_label = torch.tensor([f.mask_label for f in features]).to(device)
    mask_type = torch.tensor([f.mask_type for f in features]).to(device)
    h_input_ids = torch.tensor([f.h_input_ids for f in features], dtype=torch.long).to(device)
    h_input_mask = torch.tensor([f.h_input_mask for f in features]).to(device)
    h_incidence_matrix = torch.stack([f.h_incidence_matrix for f in features]).to(device)

    data_list = list()
    data_list.append(input_ids)
    data_list.append(input_mask)
    data_list.append(mask_position)
    data_list.append(mask_label)
    data_list.append(mask_type)
    data_list.append(input_label)
    data_list.append(h_input_ids)
    data_list.append(h_input_mask)
    data_list.append(h_incidence_matrix)

    gnn_dataset = GNNDataset(data_list)
    return gnn_dataset


def get_example_neighbor_list(neighbor_list, k, example_id, vocab_id_dict):
    sample_k = k  # sample k
    # example_ids = vocab_id_dict[example_id]
    neighbor_list = list(neighbor_list - {example_id}) if len(neighbor_list) > 0 else neighbor_list
    # overlap_list = []
    # for neighbor_id in neighbor_list:
    #     overlap_list.append(len(set(vocab_id_dict[neighbor_id]) & set(example_ids)))

    if sample_k < len(neighbor_list):
        # top_k = heapq.nlargest(k, enumerate(overlap_list), key=lambda x: x[1])
        # top_k_indices = [index for index, value in top_k]
        # sample_neighbor_list = [neighbor_list[i] for i in top_k_indices]
        sample_neighbor_list = random.sample(neighbor_list, k=k)
    else:
        sample_neighbor_list = neighbor_list
    return sample_neighbor_list


def generate_hypergraph_feature(input_ids, neighbor_list, vocab_id_dict, max_hypergraph_node_num, mask_label, mask_element, k):

    '''generate hyperedge'''
    hyperedge_id = []
    hyperedge_id.append(input_ids.copy())
    for neighbor_id in neighbor_list:
        n_input_ids = vocab_id_dict[neighbor_id]
        if mask_element:
            n_input_ids = [1 if i == mask_label else i for i in n_input_ids]
        input_ids.extend(list(set(n_input_ids) - set(input_ids)))
        hyperedge_id.append(n_input_ids)

    '''generate hypergraph incidence matrix'''
    incidence_matrix_list = []
    for i in hyperedge_id:
        node_indices = [input_ids.index(id) for id in i]
        hyperedge = torch.zeros(max_hypergraph_node_num)
        hyperedge[node_indices] = 1
        incidence_matrix_list.append(hyperedge)
    while len(incidence_matrix_list) < k + 1:  # Fill the zero vector when there are not enough hyperedges
        incidence_matrix_list.append(torch.zeros(max_hypergraph_node_num))

    incidence_matrix = torch.stack(incidence_matrix_list).t()
    # to sparse_coo_tensor
    # indices = torch.nonzero(incidence_matrix, as_tuple=False).t()
    # values = incidence_matrix[indices[0], indices[1]]
    # incidence_matrix = torch.sparse_coo_tensor(indices, values, incidence_matrix.size())

    '''feature'''
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_hypergraph_node_num:
        input_ids.append(0)  # '[PAD]' id
        input_mask.append(0)

    return input_ids, incidence_matrix, input_mask

