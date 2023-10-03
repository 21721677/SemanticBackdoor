import argparse
import os

import torch

from BackdoorModified import backdoor_modified
from BackdoorUnmodified import backdoor_unmodified
from BenignModel import benign_main
from DataAnalyzing import data_analyzing


def parse_args():
    parser = argparse.ArgumentParser(
        description="A semantic backdoor attack against Graph Convolutional Networks")
    parser.add_argument("--dataset", type=str,
                        default="AIDS", help="dataset name")
    parser.add_argument("--use_node_attr", action="store_true",
                        help="whether to use node attributes as (part of) node features")
    parser.add_argument("--n_split", type=int, default=3,
                        help="number of re-shuffling and splitting iterations")
    parser.add_argument("--k_fold", type=int, default=0,
                        help="which fold (0,1,..,n_split-1) to be used to store scoring model")

    # attack parameters
    parser.add_argument("--target", type=int, default=0,
                        help="targe label")
    parser.add_argument("--p", type=float, default=0.03,
                        help="poisoning rate")

    # GCN parameters
    parser.add_argument("--num_hidden_layer", type=int, default=1,
                        help="number of hidden layer in the GCN model")
    parser.add_argument("--num_hidden_channel", type=int, default=32,
                        help="number of hidden channel in hidden layers")

    # training parameters
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="the proportion of the dataset to include in the train split")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu",
                        help="whether to use gpu for training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="number of epochs to train")

    # other parameters
    parser.add_argument("--model_dirname", type=str, default="models",
                        help="dirname to save models")
    parser.add_argument("--log_filename", type=str, default="logs.txt")
    parser.add_argument("--result_filename", type=str, default="results.txt")

    # args = "--dataset AIDS --use_node_attr --n_split 1 --target 0 --p 0.03".split()
    # args = parser.parse_args(args)
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    if not os.path.exists("output"):
        os.mkdir("output")

    args.log_filename = f"{args.dataset}_target_{args.target}_p_{args.p}_logs.txt"
    args.result_filename = f"{args.dataset}_target_{args.target}_p_{args.p}_results.txt"

    return args


def SBAG_main(args):
    benign_main(args)
    nodes_table, trigger_node = data_analyzing(args)
    backdoor_unmodified(args, trigger_node)
    backdoor_modified(args, nodes_table, trigger_node)


if __name__ == "__main__":
    args = parse_args()
    print(args, "\n")
    SBAG_main(args)
