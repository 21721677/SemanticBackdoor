import os.path as osp
import random
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils import load_dataset, has_node, test_backdoor


def backdoor_test(args, T_nodes, trigger_node):
    dataset, num_node_attributes, num_node_labels = load_dataset(args)

    # load the backdoored model
    backdoored_model = torch.load(
        osp.join(args.model_dirname, f"{args.dataset}_backdoored_model.pt"))

    # count the average occurrence number of the trigger node
    t = T_nodes[trigger_node]["occ"]/(
        sum(T_nodes[trigger_node].values())-T_nodes[trigger_node]["occ"])

    with open(args.log_filename, "a+") as log,\
            open(args.result_filename, "a+") as result:

        output_str = f"The average occurrence number of the trigger node: {t:.2f}\n" +\
            f"Use k={int(t)} and k={int(t)+1} for testing"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)

        t = int(t)

        test_data_list = [graph for graph in dataset if (
            graph.y.item() != args.target and not has_node(graph, num_node_attributes, trigger_node))]

        # record all possible features of the target node
        feature_list = []
        if dataset.has_node_attr:
            for graph in dataset:
                for line in graph.x:
                    if line[num_node_attributes+trigger_node]:
                        feature_list.append(line[:])

        output_str = "Test the backdoored model with modified data--------------------"
        print(output_str)
        log.write(output_str+"\n")
        result.write("\n"+output_str+"\n")

        # replace k random node(s) with the target node
        if dataset.has_node_attr:
            n = len(feature_list)

            output_str = f"With node attributes, the number of total features to be tested: {n}"
            print(output_str)
            log.write(output_str+"\n")
            result.write(output_str+"\n")

            with open(f"{args.dataset}_{trigger_node}_attack.txt", "w") as wf:
                for k in [t, t+1]:
                    output_str = f"k={k}"
                    print(output_str)
                    wf.write(output_str+"\n")

                    asr_sum = 0
                    percent = 5
                    for j, feature in enumerate(feature_list):
                        if (j+1) % (n//20) == 0:
                            print(f"{percent}% finished")
                            percent += 5

                        test_data_deepcopy = deepcopy(test_data_list)
                        for i, graph in enumerate(test_data_deepcopy):
                            node_num = graph.x.shape[0]
                            # edge_num = graph.edge_index.shape[1]

                            new_graph = graph
                            node_sample = random.sample(
                                range(0, node_num), min(node_num, k))
                            for node_idx in node_sample:
                                new_graph.x[node_idx, :] = feature
                            test_data_deepcopy[i] = new_graph

                        test_loader = DataLoader(
                            test_data_deepcopy, batch_size=args.batch_size)
                        asr = test_backdoor(
                            backdoored_model, test_loader, args.target, args.device)
                        asr_sum += asr

                        wf.write(f"feature {j}".ljust(15)+f"ASR={asr:.4f}\n")

                    output_str = f"k={k} average ASR={asr_sum/len(feature_list)*100:.2f}%"
                    print(output_str+"\n")
                    wf.write(output_str+"\n"*2)
                    result.write(output_str+"\n")

                output_str = f"See details in {args.dataset}_{trigger_node}_attack.txt"
                print(output_str)
                wf.write(output_str+"\n")
        else:
            output_str = "Without node attributes:"
            print(output_str)
            log.write(output_str+"\n")
            result.write(output_str+"\n")

            for i, graph in enumerate(test_data_list):
                node_num = graph.x.shape[0]
                # edge_num = graph.edge_index.shape[1]

                new_graph = graph
                new_x = np.zeros(graph.x.shape[1])
                new_x[trigger_node] = 1
                node_sample = random.sample(
                    range(0, node_num), min(node_num, k))
                for node_idx in node_sample:
                    new_graph.x[node_idx, :] = torch.from_numpy(new_x)
                test_data_list[i] = new_graph

            test_loader = DataLoader(
                test_data_list, batch_size=args.batch_size)
            asr = test_backdoor(backdoored_model, test_loader,
                                args.target, args.device)

            output_str = f"k={k} ASR={asr*100:.2f}%"
            print(output_str)
            wf.write(output_str+"\n")
            result.write(output_str+"\n")
