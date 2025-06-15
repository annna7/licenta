import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from datasets.TimeDataset import TimeDataset

from models.GDN import GDN
from models.usad import USAD
from models.lstm_ar import LSTM_AR

from dl_train import train_gdn, train_usad, train_lstm_ar
from dl_test import test_gdn, test_usad, test_lstm_ar
from evaluate import (
    standardize_residuals,
    tune_threshold_for_best_f1,
    evaluate_threshold_performance,
    aggregate_feature_errors,
)

import os
from pathlib import Path
from datetime import datetime


class Main():
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config

        self.model_type = train_config["model_type"]  # gdn, lstm_ar or usad
        print(f"Model type: {self.model_type}")
        self.datestr = None

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)

        train_df, test_df = train_orig.copy(), test_orig.copy()
        if "attack" in train_df.columns:
            train_df = train_df.drop(columns=["attack"])

        # build graph if needed
        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)
        set_device(env_config["device"])
        self.device = get_device()

        # for GDN
        # pre‐compute fc_edge_index
        self.fc_edge_index = None
        if self.model_type == "gdn":
            raw_edge_index = build_loc_net(
                fc_struc, list(train_df.columns), feature_map=feature_map
            )
            self.fc_edge_index = torch.tensor(
                raw_edge_index, dtype=torch.long, device=self.device
            )

        self.usad_test_labels = None
        self.ar_test_labels = None

        if self.model_type == "gdn":
            train_indata = construct_data(train_df, feature_map, labels=0)
            test_indata = construct_data(test_df, feature_map, labels=test_df.attack.tolist())
            cfg = {
                "slide_win": train_config["slide_win"],
                "slide_stride": train_config["slide_stride"],
            }
            train_dataset = TimeDataset(
                train_indata, self.fc_edge_index, mode="train", config=cfg
            )
            test_dataset = TimeDataset(
                test_indata, self.fc_edge_index, mode="test", config=cfg
            )

            train_dl, val_dl = self.get_loaders(
                train_dataset,
                train_config["seed"],
                train_config["batch"],
                val_ratio=train_config["val_ratio"],
            )
            self.train_dataloader = train_dl
            self.val_dataloader = val_dl
            self.test_dataloader = DataLoader(
                test_dataset, batch_size=train_config["batch"], shuffle=False
            )
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

        # usad expects flat windows of length = (num_features × window_size) →
        elif self.model_type == "usad":
            window_size = train_config["slide_win"]
            stride = train_config["slide_stride"]
            data_arr = train_df.values

            flat_train_windows = []
            for start in range(0, data_arr.shape[0] - window_size + 1, stride):
                window = data_arr[start: start + window_size, :]  # (window_size, num_feat)
                flat_train_windows.append(window.flatten())  # (window_size * num_feat,)

            data_arr_test = test_df.drop(columns=["attack"]).values
            attack_arr = test_df["attack"].values  # shape: (T,)
            flat_test_windows = []
            test_labels = []

            for start in range(0, data_arr_test.shape[0] - window_size + 1, stride):
                window = data_arr_test[start: start + window_size, :]
                flat_test_windows.append(window.flatten())

                # label = 1 if any point in that window had attack == 1
                window_labels = attack_arr[start: start + window_size]
                test_labels.append(int(window_labels.max() > 0))

            self.usad_test_labels = np.array(test_labels, dtype=np.int64)

            train_tensor = torch.tensor(flat_train_windows, dtype=torch.float32)
            train_labels = torch.zeros(len(flat_train_windows), dtype=torch.long)  # all‐zeros
            test_tensor = torch.tensor(flat_test_windows, dtype=torch.float32)

            train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)  # (x_flat,)
            test_labels_tensor = torch.tensor(self.usad_test_labels, dtype=torch.long)
            test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels_tensor)

            val_size = int(len(train_dataset) * train_config["val_ratio"])
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size]
            )

            self.train_dataloader = DataLoader(
                train_subset, batch_size=train_config["batch"], shuffle=True
            )
            self.val_dataloader = DataLoader(
                val_subset, batch_size=train_config["batch"], shuffle=False
            )
            self.test_dataloader = DataLoader(
                test_dataset, batch_size=train_config["batch"], shuffle=False
            )
            self.train_dataset = train_dataset

        # — LSTM_AR expects (window, next_step) pairs →
        elif self.model_type == "lstm_ar":
            window_size = train_config["slide_win"]
            data_arr = train_df.values

            train_windows = []
            train_nexts = []
            for start in range(0, data_arr.shape[0] - window_size):
                win = data_arr[start: start + window_size, :]
                nxt = data_arr[start + window_size, :]
                train_windows.append(win)
                train_nexts.append(nxt)

            windows_tensor = torch.tensor(train_windows, dtype=torch.float32)  # (N, window_size, num_features)
            nexts_tensor = torch.tensor(train_nexts, dtype=torch.float32)  # (N, num_features)

            # dummy zero‐labels so that DataLoader yields (window, next_step, label)
            dummy_train_labels = torch.zeros(len(train_windows), dtype=torch.long)

            full_dataset = torch.utils.data.TensorDataset(
                windows_tensor,
                nexts_tensor,
                dummy_train_labels
            )

            val_size = int(len(full_dataset) * train_config["val_ratio"])
            train_size = len(full_dataset) - val_size
            train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

            self.train_dataloader = DataLoader(
                train_subset, batch_size=train_config["batch"], shuffle=True
            )
            self.val_dataloader = DataLoader(
                val_subset, batch_size=train_config["batch"], shuffle=False
            )

            test_df_no_attack = test_df.drop(columns=["attack"])
            data_arr_test = test_df_no_attack.values
            attack_arr_test = test_df["attack"].values

            test_windows = []
            test_nexts = []
            test_labels = []
            for start in range(0, data_arr_test.shape[0] - window_size):
                win = data_arr_test[start: start + window_size, :]
                nxt = data_arr_test[start + window_size, :]
                test_windows.append(win)
                test_nexts.append(nxt)
                test_labels.append(int(attack_arr_test[start + window_size]))

            test_win_tensor = torch.tensor(test_windows, dtype=torch.float32)
            test_next_tensor = torch.tensor(test_nexts, dtype=torch.float32)
            test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

            test_dataset = torch.utils.data.TensorDataset(
                test_win_tensor,
                test_next_tensor,
                test_labels_tensor
            )
            self.test_dataloader = DataLoader(
                test_dataset, batch_size=train_config["batch"], shuffle=False
            )

            # Keep references
            self.train_dataset = full_dataset
            self.ar_test_labels = np.array(test_labels, dtype=np.int64)


        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        if self.model_type == "gdn":
            node_num = len(feature_map)  # number of graph nodes (sensors)
            self.model = GDN(
                edge_index_sets=[self.fc_edge_index],
                node_num=node_num,
                dim=train_config["dim"],
                out_layer_inter_dim=train_config["out_layer_inter_dim"],
                input_dim=train_config["slide_win"],
                out_layer_num=train_config["out_layer_num"],
                topk=train_config["topk"],
            ).to(self.device)

        elif self.model_type == "usad":
            window_size_flat = train_config["slide_win"] * train_df.shape[1]
            latent_size = train_config["latent_size"]
            self.model = USAD(window_size=window_size_flat, latent_size=latent_size).to(
                self.device
            )

        elif self.model_type == "lstm_ar":
            num_features = train_df.shape[1]
            hidden_size = train_config["hidden_size"]
            num_layers = train_config["num_layers"]
            dropout = train_config["dropout"]
            self.model = LSTM_AR(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            ).to(self.device)

        # Keep feature_map around if needed for GDN evaluation
        self.feature_map = feature_map

    def run(self):
        # If load_model_path is provided, skip training
        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            model_save_path = self.get_save_path()[0]

            # Dispatch to the appropriate training function based on self.model_type
            if self.model_type == "gdn":
                self.train_log = train_gdn(
                    model=self.model,
                    save_path=model_save_path,
                    config=self.train_config,
                    train_dataloader=self.train_dataloader,
                    val_dataloader=self.val_dataloader,
                    feature_map=self.feature_map,
                    test_dataloader=self.test_dataloader,
                    test_dataset=getattr(self, "test_dataset", None),
                    train_dataset=self.train_dataset,
                    dataset_name=self.env_config["dataset"],
                )

            elif self.model_type == "usad":
                self.train_log = train_usad(
                    model=self.model,
                    save_path=model_save_path,
                    config=self.train_config,
                    train_dataloader=self.train_dataloader,
                    val_dataloader=self.val_dataloader,
                )

            elif self.model_type == "lstm_ar":
                self.train_log = train_lstm_ar(
                    model=self.model,
                    save_path=model_save_path,
                    config=self.train_config,
                    train_dataloader=self.train_dataloader,
                    val_dataloader=self.val_dataloader,
                )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        if self.model_type == "gdn":
            _, test_result = test_gdn(best_model, self.test_dataloader)
            _, val_result = test_gdn(best_model, self.val_dataloader)
            self.get_score(test_result, val_result)

        elif self.model_type == "usad":
            _, test_result = test_usad(
                best_model,
                self.test_dataloader,
                alpha=self.train_config["alpha"],
                beta=self.train_config["beta"],
            )
            _, val_result = test_usad(best_model, self.val_dataloader, alpha=self.train_config['alpha'],
                                      beta=self.train_config['beta'])
            self.get_score(test_result, val_result)

        elif self.model_type == "lstm_ar":
            _, test_result = test_lstm_ar(best_model, self.test_dataloader)
            _, val_result = test_lstm_ar(best_model, self.val_dataloader)
            self.get_score(test_result, val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len:]]
        )
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index: val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)
        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = aggregate_feature_errors(test_result, val_result)
        dataset_name = self.env_config["dataset"]
        subdir = f"scores_output/{dataset_name}_1"
        os.makedirs(subdir, exist_ok=True)

        np.save(os.path.join(subdir, "test_scores.npy"), test_scores)
        np.save(os.path.join(subdir, "normal_scores.npy"), normal_scores)
        np.save(os.path.join(subdir, "test_labels.npy"), test_labels)

        print(f"[Saved] {subdir}/test_scores.npy")
        print(f"[Saved] {subdir}/normal_scores.npy")
        print(f"[Saved] {subdir}/test_labels.npy")

        top1_best_info = tune_threshold_for_best_f1(test_scores, test_labels, topk=1)
        top1_val_info = evaluate_threshold_performance(test_scores, normal_scores, test_labels, topk=1)

        print("=========================** Result **============================\n")
        if self.env_config["report"] == "best":
            info = top1_best_info
        else:
            info = top1_val_info

        print(f"F1 score: {info[0]:.4f}")
        print(f"precision: {info[1]:.4f}")
        print(f"recall: {info[2]:.4f}\n")

    def get_save_path(self, feature_name=""):
        dir_path = self.env_config["save_path"]
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m|%d-%H:%M:%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]
        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model_type",
        dest="model_type",  # maps to args.model_type
        type=str,
        required=True,
        choices=["gdn", "usad", "lstm_ar"],
        help="Which model to train: gdn | usad | lstm_ar",
    )
    parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=100)
    parser.add_argument("-slide_win", help="slide_win", type=int, default=15)
    parser.add_argument("-dim", help="dimension (for GDN)", type=int, default=64)
    parser.add_argument(
        "-hidden_size", help="hidden size (for LSTM_AR)", type=int, default=64
    )
    parser.add_argument("-num_layers", help="number of LSTM layers", type=int, default=2)
    parser.add_argument("-dropout", help="dropout for LSTM_AR", type=float, default=0.0)
    parser.add_argument(
        "-latent_size", help="latent size (for USAD)", type=int, default=32
    )
    parser.add_argument("-slide_stride", help="slide_stride", type=int, default=5)
    parser.add_argument(
        "-save_path_pattern", help="save path pattern", type=str, default=""
    )
    parser.add_argument("-dataset", help="wadi / swat", type=str, default="wadi")
    parser.add_argument("-device", help="cuda / cpu", type=str, default="cuda")
    parser.add_argument("-random_seed", help="random seed", type=int, default=0)
    parser.add_argument(
        "-comment", help="experiment comment", type=str, default=""
    )
    parser.add_argument(
        "-out_layer_num", help="outlayer num (GDN)", type=int, default=1
    )
    parser.add_argument(
        "-out_layer_inter_dim", help="out_layer_inter_dim (GDN)", type=int, default=256
    )
    parser.add_argument("-decay", help="decay (GDN)", type=float, default=0)
    parser.add_argument("-val_ratio", help="val ratio", type=float, default=0.1)
    parser.add_argument("-topk", help="topk num (GDN)", type=int, default=20)
    parser.add_argument(
        "-alpha", help="alpha (for USAD scoring)", type=float, default=0.5
    )
    parser.add_argument(
        "-beta", help="beta (for USAD scoring)", type=float, default=0.5
    )
    parser.add_argument(
        "-report", help="best / val (for GDN)", type=str, default="best"
    )
    parser.add_argument(
        "-load_model_path", help="trained model path", type=str, default=""
    )

    args = parser.parse_args()

    # Seed everything
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "model_type": args.model_type,  # store the string flag here
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "slide_stride": args.slide_stride,
        "comment": args.comment,
        "seed": args.random_seed,
        "out_layer_num": args.out_layer_num,
        "out_layer_inter_dim": args.out_layer_inter_dim,
        "decay": args.decay,
        "val_ratio": args.val_ratio,
        "topk": args.topk,
        "dim": args.dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "latent_size": args.latent_size,
        "alpha": args.alpha,
        "beta": args.beta,
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "report": args.report,
        "device": args.device,
        "load_model_path": args.load_model_path,
    }
    print('Device', end=' ')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    main = Main(train_config, env_config, debug=False)
    main.run()
    print(f'Ran pipeline with {args.model_type}; {args.dataset}')