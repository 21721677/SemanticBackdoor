from copy import copy, deepcopy
import os
import os.path as osp

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from utils import load_dataset, GCN, train_model, test_model, test_backdoor,\
    has_node, predict_sample, modify_features


def backdoor_unmodified(args, trigger_node):
    dataset, num_node_attributes, num_node_labels = load_dataset(args)
    labels = [graph.y.item()
              for graph in dataset] if dataset.num_classes == 2 else None

    # split the dataset into random train and test subsets
    train_data, test_data = train_test_split(
        dataset, train_size=args.train_size, stratify=labels)

    # load the scoring model
    scoring_model = torch.load(
        osp.join(args.model_dirname, f"{args.dataset}_scoring_model.pt"))

    POISONING_DATA_DIRNAME = "poisoning_data"
    if not osp.exists(POISONING_DATA_DIRNAME):
        os.mkdir(POISONING_DATA_DIRNAME)
    poisoning_data_dir = osp.join(POISONING_DATA_DIRNAME, args.dataset)
    if not osp.exists(poisoning_data_dir):
        os.mkdir(poisoning_data_dir)

    poisoning_data_list = []  # poisoning samples for training backdoored model
    benign_data_list = []  # benign samples for benign classification test
    backdoor_data_list = []  # original samples for backdoor classification test

    poisoning_num = int(len(dataset)*args.p)

    with open(args.log_filename, "a+") as log,\
            open(args.result_filename, "a+") as result:

        output_str = "Select top-k candidate samples and relabel them--------------------"
        print(output_str)
        log.write(output_str+"\n")

        candidate_data_list = []  # candidate samples for poisoning
        for i, graph in enumerate(train_data):
            if has_node(graph, num_node_attributes, trigger_node) and graph.y.item() != args.target:
                score_one = predict_sample(
                    scoring_model, copy(graph), args.device)
                new_graph = modify_features(
                    deepcopy(graph), num_node_attributes, trigger_node)
                score_two = predict_sample(
                    scoring_model, new_graph, args.device)

                diff = abs(score_one-score_two)
                candidate_data_list.append((graph, diff))
            else:
                poisoning_data_list.append(graph)

        # sort the array by diff and select top-k samples for poisoning
        candidate_data_list.sort(key=lambda x: x[1], reverse=True)
        poison_data = [x[0] for x in candidate_data_list]
        poisoning_num = min(poisoning_num, len(candidate_data_list))

        # relabel top-k samples
        for i in range(0, poisoning_num):
            # print(f"original class: {poison_data[i].y.item()}")
            poison_data[i].y = torch.tensor([args.target])

        poisoning_data_list.extend(poison_data)

        # divide the testing set into two parts
        for i, graph in enumerate(test_data):
            if has_node(graph, num_node_attributes, trigger_node) and graph.y.item() != args.target:
                backdoor_data_list.append(graph)
            else:
                benign_data_list.append(graph)

        p_rate = poisoning_num/len(dataset)
        output_str = f"The number of samples for poisoning: {poisoning_num}, poison rate: {p_rate*100:.1f}%"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)

        torch.save(poisoning_data_list, osp.join(
            poisoning_data_dir, "poisoning_data.pt"))
        torch.save(benign_data_list, osp.join(
            poisoning_data_dir, "benign_data.pt"))
        torch.save(backdoor_data_list, osp.join(
            poisoning_data_dir, "backdoor_data.pt"))

        poisoning_train_loader = DataLoader(
            poisoning_data_list, batch_size=args.batch_size, shuffle=True)
        benign_test_loader = DataLoader(
            benign_data_list, batch_size=args.batch_size)
        backdoor_test_loader = DataLoader(
            backdoor_data_list, batch_size=args.batch_size)

        backdoored_model = GCN(args.num_hidden_layer, dataset.num_node_features,
                               args.num_hidden_channel, dataset.num_classes,
                               args.device).to(args.device)
        optimizer = torch.optim.Adam([
            dict(params=backdoored_model.conv_in.parameters(), weight_decay=5e-4),
            dict(params=backdoored_model.conv_out.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

        output_str = "Start training the backdoored model--------------------"
        print(output_str)
        log.write(output_str+"\n")

        for epoch in range(0, args.max_epoch):
            train_loss = train_model(
                backdoored_model, poisoning_train_loader, optimizer, args.device)
            if epoch % 10 == 0:
                output_str = f"Epoch: {epoch:03d}, Train loss: {train_loss/len(train_data):.4f}"
                print(output_str)
                log.write(output_str+"\n")

        output_str = "Finish training the backdoored model--------------------"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)

        output_str = "Test the backdoored model with unmodified data:"
        print(output_str)
        log.write(output_str+"\n")
        result.write(output_str+"\n")

        benign_acc = test_model(
            backdoored_model, benign_test_loader, args.device)
        backdoor_asr = test_backdoor(
            backdoored_model, backdoor_test_loader, args.target, args.device)

        output_str = f"Normal Acc: {benign_acc*100:.2f}%\nBackdoor ASR: {backdoor_asr*100:.2f}%"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)
        result.write(output_str+"\n")

        torch.save(backdoored_model, osp.join(
            args.model_dirname, f"{args.dataset}_backdoored_model.pt"))
