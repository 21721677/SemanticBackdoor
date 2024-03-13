import argparse
import os

import torch
from torch_geometric import seed_everything

from BackdoorModified import backdoor_modified
from BackdoorUnmodified import backdoor_unmodified
from CleanModel import clean_main
from DataAnalyzing import data_analyzing


def parse_args():
    parser = argparse.ArgumentParser(
        description="A semantic backdoor attack against Graph Convolutional Networks")
    parser.add_argument("--dataset", type=str,
                        default="AIDS", help="dataset name")
    parser.add_argument("--use_node_attr", action="store_true",
                        help="whether to use node attributes as (part of) node features")
    parser.add_argument("--n_split", type=int, default=1,
                        help="number of re-shuffling and splitting iterations before training the clean model")
    parser.add_argument("--k_fold", type=int, default=0,
                        help="which fold (0,1,..,n_split-1) to be used to store the scoring model")

    # attack parameters
    parser.add_argument("--target", type=int, default=0,
                        help="targe class")
    parser.add_argument("--p", type=float, default=0.03,
                        help="poisoning rate")

    # model parameters
    parser.add_argument("--model", type=str, choices=["GIN", "GCN"], default="GCN")
    parser.add_argument("--num_hidden_layer", type=int, default=1,
                        help="number of hidden layers in the target model")
    parser.add_argument("--num_hidden_channel", type=int, default=32,
                        help="number of hidden channels in hidden layers")

    # training parameters
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="the proportion of the dataset to include in the train split")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="number of epochs to train")

    # other parameters
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model_dirname", type=str, default="models",
                        help="dirname to save models")
    parser.add_argument("--log_filename", type=str, default="logs.txt")
    parser.add_argument("--result_filename", type=str, default="results.txt")

    # args = "--dataset AIDS --use_node_attr --target 0 --p 0.03 --model GCN".split()
    # args = "--dataset TWITTER-Real-Graph-Partial --target 1 --p 0.01 --model GCN".split()
    # args = parser.parse_args(args)
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    if args.seed:
        seed_everything(args.seed)

    if not os.path.exists("output"):
        os.mkdir("output")

    args.log_filename = f"{args.dataset}_target_{args.target}_p_{args.p}_{args.model}_logs.txt"
    args.result_filename = f"{args.dataset}_target_{args.target}_p_{args.p}_{args.model}_results.txt"

    return args


def SBAG_main(args):
    clean_main(args)
    nodes_table, trigger_node = backdoor_unmodified(args)
    backdoor_modified(args, nodes_table, trigger_node)


if __name__ == "__main__":
    args = parse_args()
    print(args, "\n")
    SBAG_main(args)
