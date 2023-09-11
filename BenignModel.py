import os
import os.path as osp

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

from utils import load_dataset, GCN, train_model, test_model


def benign_main(args):
    dataset, *rest = load_dataset(args)

    print(f"Total samples: {len(dataset)}")
    for label in range(0, dataset.num_classes):
        print(f"class {label}:",
              sum([graph.y.item() == label for graph in dataset]))
    print()

    MODEL_DIRNAME = args.model_dirname
    if not osp.exists(MODEL_DIRNAME):
        os.mkdir(MODEL_DIRNAME)

    # stratified split data in train/test sets
    labels = [graph.y.item()
              for graph in dataset] if dataset.num_classes == 2 else None
    sss = StratifiedShuffleSplit(
        n_splits=args.n_split, train_size=args.train_size)

    benign_acc = []
    with open(args.log_filename, "a+") as log,\
            open(args.result_filename, "a+") as result:

        output_str = "Start training the benign model--------------------"
        print(output_str)
        log.write(output_str+"\n")

        for i, (train_index, test_index) in enumerate(sss.split(dataset, labels)):
            output_str = f"{'='*20}Fold {i}{'='*20}"
            print(output_str)
            log.write(output_str+"\n")

            train_data = [dataset[i] for i in train_index]
            test_data = [dataset[i] for i in test_index]

            train_loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=args.batch_size)

            benign_model = GCN(args.num_hidden_layer, dataset.num_node_features,
                               args.num_hidden_channel, dataset.num_classes,
                               args.device).to(args.device)
            optimizer = torch.optim.Adam([
                dict(params=benign_model.conv_in.parameters(), weight_decay=5e-4),
                dict(params=benign_model.conv_out.parameters(), weight_decay=0)
            ], lr=args.lr)  # only perform weight-decay on first convolution.

            best_test_epoch, best_test_acc = 0, 0
            for epoch in range(0, args.max_epoch):
                train_loss = train_model(
                    benign_model, train_loader, optimizer, args.device)
                test_acc = test_model(benign_model, test_loader, args.device)
                if test_acc > best_test_acc:
                    best_test_epoch, best_test_acc = epoch, test_acc
                    output_str = f"Best test epoch: {best_test_epoch}, best test accuracy: {best_test_acc*100:.2f}%"
                    print(output_str)
                    log.write(output_str+"\n")

                    # save the scoring model
                    if i == args.k_fold:
                        torch.save(benign_model, osp.join(
                            MODEL_DIRNAME, f"{args.dataset}_scoring_model.pt"))
                if epoch % 10 == 0:
                    output_str = f"Epoch: {epoch:03d}, train loss: {train_loss/len(train_data):.4f}"
                    print(output_str)
                    log.write(output_str+"\n")

            test_acc = test_model(benign_model, test_loader, args.device)
            benign_acc.append(test_acc)
            output_str = f"Max epoch={args.max_epoch}, test accuracy={test_acc*100:.2f}%"
            print(output_str)
            log.write(output_str+"\n")

        output_str = "Finish training the benign model--------------------"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)

        ave_benign_acc = sum(benign_acc)/len(benign_acc)
        output_str = f"Average clean accuracy: {ave_benign_acc*100:.2f}%"
        print(output_str+"\n")
        log.write(output_str+"\n"*2)
        result.write(output_str+"\n"*2)
