import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import TUDataset


def data_analyzing(args):
    dataset_name, target, p = args.dataset, args.target, args.p
    dataset = TUDataset(
        root="temp", name=dataset_name, use_node_attr=False)

    with open(args.log_filename, "a+") as log,\
            open(args.result_filename, "a+") as result:

        output_str = "Start analyzing the dataset--------------------"
        print(output_str)
        log.write(output_str+"\n")

        n = len(dataset)
        num_classes = dataset.num_classes
        num_node_labels = dataset.num_node_labels
        occ_num = np.zeros(num_node_labels, dtype=int)
        nodes_table = defaultdict(
            lambda: {x: 0 for x in range(0, num_classes)})

        for graph in tqdm(dataset, desc=dataset_name, file=sys.stdout):
            # count the occurrence number of each node
            sum_array = graph.x.sum(axis=0).numpy().astype(int)
            occ_num += sum_array

            # count the corresponding graph labels
            for node, num in enumerate(sum_array):
                if num > 0:
                    nodes_table[node][graph.y.item()] += 1

        for node, num in enumerate(occ_num):
            nodes_table[node]["occ"] = num

        # write results
        with open(f"{dataset_name}_analyze.txt", "w") as wf:
            for node in range(0, num_node_labels):
                wf.write(f"node {node}: {nodes_table[node]['occ']}".ljust(15))
                for label in range(0, num_classes):
                    wf.write(
                        f"class {label}: {nodes_table[node][label]}".ljust(15))
                wf.write("\n")

        output_str = f"Finish analyzing the dataset, see results in {dataset_name}_analyze.txt"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)

        poisoning_num = int(n*p)
        trigger_node, min_diff = -1, float("inf")
        for node in nodes_table:
            ava_num = sum(nodes_table[node][label] for label in range(
                0, num_classes))-nodes_table[node][target]
            diff = abs(ava_num-poisoning_num)
            if diff < min_diff:
                trigger_node, min_diff = node, diff

        output_str = f"Select node {trigger_node} as the trigger node"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)
        result.write(output_str+"\n"*2)
        return nodes_table, trigger_node
