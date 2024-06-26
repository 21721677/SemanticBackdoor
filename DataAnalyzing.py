import os.path as osp
import sys
from collections import defaultdict

import numpy as np
from torch_geometric.datasets import TUDataset
from tqdm import tqdm


def data_analyzing(args, train_data, poisoning_num, num_node_attributes, num_node_labels, num_classes):
    dataset_name, target = args.dataset, args.target

    with open(osp.join("output", args.log_filename), "a+") as log, \
            open(osp.join("output", args.result_filename), "a+") as result:

        output_str = "Start analyzing the dataset--------------------"
        print(output_str)
        log.write(output_str + "\n")

        occ_num = np.zeros(num_node_labels, dtype=int)
        nodes_table = defaultdict(
            lambda: {"occ": 0, **{x: 0 for x in range(0, num_classes)}})

        for graph in tqdm(train_data, desc=dataset_name, file=sys.stdout):
            # count the occurrence number of each node
            sum_array = graph.x.sum(axis=0).numpy().astype(int)
            occ_num += sum_array[num_node_attributes:]

            # count the corresponding graph labels
            for node, num in enumerate(sum_array):
                if num > 0:
                    nodes_table[node][graph.y.item()] += 1

        for node, num in enumerate(occ_num):
            nodes_table[node]["occ"] = num

        # write results
        with open(osp.join("output", f"{dataset_name}_analyze.txt"), "w") as wf:
            for node in range(0, num_node_labels):
                wf.write(f"node {node}: {nodes_table[node]['occ']}".ljust(15))
                for label in range(0, num_classes):
                    wf.write(
                        f"class {label}: {nodes_table[node][label]}".ljust(15))
                wf.write("\n")

        output_str = f"Finish analyzing the dataset, see results in output/{dataset_name}_analyze.txt"
        print(output_str + "\n")
        log.write(output_str + "\n" * 2)

        trigger_node, min_diff = -1, float("inf")
        for node in nodes_table:
            ava_num = sum(nodes_table[node][label] for label in range(
                0, num_classes)) - nodes_table[node][target]
            diff = abs(ava_num - poisoning_num)
            if diff < min_diff:
                trigger_node, min_diff = node, diff

        output_str = f"Select node {trigger_node} as the trigger node"
        print(output_str + "\n")
        log.write(output_str + "\n" * 2)
        result.write(output_str + "\n" * 2)
        return nodes_table, trigger_node
