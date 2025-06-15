import argparse
import itertools
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from baseline_models import (
    compute_pot_threshold,
    run_ocsvm,
    run_lof,
    run_var,
    run_arma,
    windowize,
    compute_metrics
)


def downsample_data(data, labels, target_length):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    num_samples, num_features = data.shape

    if num_samples < target_length:
        raise ValueError(
            f"Cannot downsample to {target_length}; only have {num_samples} samples available."
        )

    downsample_factor = num_samples // target_length
    usable = downsample_factor * target_length
    truncated_data = data[:usable]
    truncated_labels = labels[:usable]

    reshaped_data = truncated_data.reshape(target_length, downsample_factor, num_features)
    reshaped_labels = truncated_labels.reshape(target_length, downsample_factor)

    downsampled_data = np.median(reshaped_data, axis=1)
    downsampled_labels = np.max(reshaped_labels, axis=1).astype(int)

    return downsampled_data.tolist(), downsampled_labels.tolist()


def load_data_from_csv(data_dir: str, dataset: str):
    base = Path(data_dir) / dataset

    train_csv = base / "train.csv"
    test_csv = base / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Cannot find: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Cannot find: {test_csv}")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if "attack" not in df_train.columns:
        raise ValueError(f"'attack' column missing in {train_csv}")
    if "attack" not in df_test.columns:
        raise ValueError(f"'attack' column missing in {test_csv}")

    train_labels = df_train["attack"].astype(int).to_numpy()
    test_labels = df_test["attack"].astype(int).to_numpy()

    df_train = df_train.drop(columns=["attack"])
    df_test = df_test.drop(columns=["attack"])

    train_arr = df_train.to_numpy(dtype=float)
    test_arr = df_test.to_numpy(dtype=float)

    return train_arr, train_labels, test_arr, test_labels


def normalize_data(train: np.ndarray, test: np.ndarray):
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)
    return train_norm, test_norm


def sliding_window_labels(labels: np.ndarray, window_size: int):
    T = len(labels)
    if T <= window_size:
        return np.zeros((0,), dtype=int)

    windowed = np.zeros((T - window_size,), dtype=int)
    for i in range(T - window_size):
        windowed[i] = int(np.any(labels[i: i + window_size]))
    return windowed


def make_results_folder(path: str):
    os.makedirs(path, exist_ok=True)


def save_results_csv(
        out_folder: str,
        dataset: str,
        baseline_name: str,
        downsample: int,
        pot_thresh: float,
        prec: float,
        rec: float,
        f1: float
):
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "downsample": downsample,
        "pot_threshold": pot_thresh,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }])

    out_filename = f"{dataset}_{baseline_name}_{downsample if downsample is not None else 'None'}.csv"
    out_path = Path(out_folder) / out_filename
    df.to_csv(out_path, index=False)
    print(f"[Saved summary] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root folder containing subfolders per dataset.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset folder name (e.g. 'SMD', 'SMAP', etc.).")
    parser.add_argument("--window_size", type=int, default=100,
                        help="Sliding window size for all baselines.")
    parser.add_argument("--baselines", type=str, default="lof",
                        help="Comma‐separated list of baselines to run.")
    parser.add_argument("--ocsvm_kernels", type=str, default="linear,poly,rbf,tophat",
                        help="Kernels for OC‐SVM: comma‐separated.")
    parser.add_argument("--lof_neighbors", type=int, default=20,
                        help="Number of neighbors for LOF.")
    parser.add_argument("--eis_models", type=int, default=5,
                        help="Number of IsolationForest estimators for EIS.")
    parser.add_argument("--eis_subsample", type=int, default=None,
                        help="Subsample size for each IF in EIS (None = full train).")
    parser.add_argument("--arma_p", type=int, default=5,
                        help="AR order p for ARMA baseline.")
    parser.add_argument("--var_lag", type=int, default=5,
                        help="Lag order for VAR baseline.")
    parser.add_argument("--nystroem_components", type=int, default=None,
                        help="If set, use Nystroem kernel approximation for OC‐SVM.")
    parser.add_argument("--ocsvm_subsample", type=int, default=None,
                        help="If set, randomly subsample this many train points before fitting OC‐SVM.")
    parser.add_argument("--tophat_radius", type=float, default=None,
                        help="If using tophat kernel, force a radius (otherwise estimated).")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Folder to save per‐baseline CSVs.")
    parser.add_argument("--downsample", type=int, default=None,
                        help="If set, reduce both train/test to this many rows via median‐binning (and max on 'attack').")
    args = parser.parse_args()

    train_raw, train_labels_raw, test_raw, labels_raw = load_data_from_csv(
        args.data_dir, args.dataset
    )

    if args.downsample is not None:
        dummy_train_labels = np.zeros(train_raw.shape[0], dtype=int)
        down_train_data, _ = downsample_data(train_raw, dummy_train_labels, args.downsample)
        train_raw = np.array(down_train_data, dtype=float)

        down_test_data, down_test_labels = downsample_data(test_raw, labels_raw, args.downsample)
        test_raw = np.array(down_test_data, dtype=float)
        labels_raw = np.array(down_test_labels, dtype=int)

        print(f"[Downsampled] train {train_raw.shape}, test {test_raw.shape}, labels {labels_raw.shape}")
    else:
        print("[No downsampling]")

    train_norm, test_norm = normalize_data(train_raw, test_raw)

    W = args.window_size
    X_train_mat = windowize(train_norm, W)  # shape: (T_train - W, W*D)
    X_test_mat = windowize(test_norm, W)  # shape: (T_test - W,  W*D)

    y_test_win = sliding_window_labels(labels_raw, W)
    print(f"[Windowized] X_train {X_train_mat.shape}, X_test {X_test_mat.shape}, y_test_win {y_test_win.shape}")

    make_results_folder(args.results_dir)
    baselines_to_run = [b.strip() for b in args.baselines.split(",")]

    for baseline in baselines_to_run:
        print("=== Running baseline:", baseline)

        if baseline == "ocsvm":
            for ker in args.ocsvm_kernels.split(","):
                ker = ker.strip()
                name = f"ocsvm_{ker}"
                print(f"— OC‐SVM (kernel={ker}) —")

                start_time = time.time()
                # Fit and score
                scores_train = run_ocsvm(
                    X_train_mat, X_train_mat,
                    kernel=ker,
                    # pca_components=750,
                    subsample=args.ocsvm_subsample,
                    nystroem_components=args.nystroem_components,
                    tophat_radius=args.tophat_radius
                )
                print('Done train')
                scores_test = run_ocsvm(
                    X_train_mat, X_test_mat,
                    kernel=ker,
                    # pca_components=750,
                    subsample=args.ocsvm_subsample,
                    nystroem_components=args.nystroem_components,
                    tophat_radius=args.tophat_radius
                )
                pot_th = compute_pot_threshold(scores_train, scores_test)
                preds_test = (scores_test >= pot_th).astype(int)
                prec, rec, f1 = compute_metrics(y_test_win, preds_test)
                elapsed = time.time() - start_time

                print(f"    → Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
                print(f"    — OC‐SVM ({ker}) took {elapsed:.2f} seconds")

                # Save summary CSV (only one row)
                save_results_csv(
                    args.results_dir,
                    args.dataset,
                    name,
                    downsample=args.downsample if args.downsample is not None else -1,
                    pot_thresh=pot_th,
                    prec=prec,
                    rec=rec,
                    f1=f1
                )

        elif baseline == "iso":
            n_estimators_grid = [50, 100, 200, 500, 750]
            max_samples_grid = [256, 512, 1024, 2048]
            best_f1 = -1.0
            best_cfg = None
            results = {}  # map (n_est, max_samp) -> F1

            # Grid search
            for n_est, max_samp in itertools.product(n_estimators_grid, max_samples_grid):
                print(f"\nTraining IF (n_estimators={n_est}, max_samples={max_samp})")
                start_time = time.time()

                iso = IsolationForest(
                    n_estimators=n_est,
                    max_samples=max_samp,
                    n_jobs=-1,
                    random_state=42
                )
                iso.fit(X_train_mat)

                # invert decision_function for anomaly score
                scores_train = -iso.decision_function(X_train_mat)
                scores_test = -iso.decision_function(X_test_mat)

                pot_th = compute_pot_threshold(scores_train, scores_test)
                preds_test = (scores_test >= pot_th).astype(int)
                prec, rec, f1 = compute_metrics(y_test_win, preds_test)
                elapsed = time.time() - start_time

                print(f"  → Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
                print(f"  → Time: {elapsed:.2f}s")

                model_name = f"iso_n{n_est}_ms{max_samp}"
                save_results_csv(
                    args.results_dir,
                    args.dataset,
                    model_name,
                    downsample=(args.downsample if args.downsample is not None else -1),
                    pot_thresh=pot_th,
                    prec=prec,
                    rec=rec,
                    f1=f1
                )

                results[(n_est, max_samp)] = f1
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = (n_est, max_samp)

            print(f"\nBest IF config: n_estimators={best_cfg[0]}, max_samples={best_cfg[1]} → F1={best_f1:.4f}")

            # Build a DataFrame of F1 scores
            df = pd.DataFrame(
                [[results[(n_est, ms)] for ms in max_samples_grid]
                 for n_est in n_estimators_grid],
                index=n_estimators_grid,
                columns=max_samples_grid
            )

            # Display the grid as a table (optional)
            print("\nF1 Score Grid:")
            print(df)

            # Plot heatmap
            plt.figure()
            plt.imshow(df.values, aspect='auto')
            plt.xticks(ticks=range(len(max_samples_grid)), labels=max_samples_grid)
            plt.yticks(ticks=range(len(n_estimators_grid)), labels=n_estimators_grid)
            plt.xlabel('max_samples')
            plt.ylabel('n_estimators')
            plt.title('IsolationForest F1 Score Grid Heatmap')
            plt.colorbar(label='F1 Score')
            plt.tight_layout()

            out_path = os.path.join(args.results_dir, "iso_forest_heatmap_wadi.png")
            plt.savefig(out_path, dpi=150)  # you can adjust dpi as needed
            plt.close()
            print(f"Heatmap saved to {out_path}")
        elif baseline == "lof":
            neighbors_grid = [10, 30, 50]
            best_f1 = -1.0
            best_k = None
            results = {}  # k -> F1

            for k in neighbors_grid:
                print(f"\n— LOF (n_neighbors={k}) —")
                start_time = time.time()

                scores_train = run_lof(X_train_mat, X_train_mat, n_neighbors=k)
                scores_test = run_lof(X_train_mat, X_test_mat, n_neighbors=k)

                pot_th = compute_pot_threshold(scores_train, scores_test)
                preds_test = (scores_test >= pot_th).astype(int)
                prec, rec, f1 = compute_metrics(y_test_win, preds_test)
                elapsed = time.time() - start_time

                print(f"    → Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
                print(f"    — LOF k={k} took {elapsed:.2f}s")

                save_results_csv(
                    args.results_dir,
                    args.dataset,
                    f"lof_k{k}",
                    downsample=(args.downsample if args.downsample is not None else -1),
                    pot_thresh=pot_th,
                    prec=prec,
                    rec=rec,
                    f1=f1
                )

                results[k] = f1
                if f1 > best_f1:
                    best_f1 = f1
                    best_k = k

            print(f"\nBest LOF config: n_neighbors={best_k} → F1={best_f1:.4f}")

            df = pd.DataFrame.from_dict(results, orient='index', columns=['F1'])
            df.index.name = 'n_neighbors'
            print("\nF1 Score by n_neighbors:")
            print(df)

            plt.figure()
            plt.plot(neighbors_grid, [results[k] for k in neighbors_grid], marker='o')
            plt.xlabel('n_neighbors')
            plt.ylabel('F1 Score')
            plt.title(f'LOF F1 vs n_neighbors ({args.dataset})')
            plt.tight_layout()
            out_path = os.path.join(args.results_dir, f"lof_f1_{args.dataset}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Plot saved to {out_path}")

        elif baseline == "var":
            from statsmodels.tsa.api import VAR
            from sklearn.decomposition import PCA
            import numpy as np

            pca = PCA(svd_solver="full")
            pca.fit(train_norm)

            eps = 1e-2
            ev = pca.explained_variance_
            n_keep = np.sum(ev > eps)
            print(f"PCA: keeping {n_keep}/{len(ev)} components (drop ev <= {eps})")

            pca = PCA(n_components=n_keep, svd_solver="full")
            train_pca = pca.fit_transform(train_norm)
            test_pca = pca.transform(test_norm)

            maxlags = 15
            print(f"— VAR with automatic lag selection (maxlags=3{maxlags}) across AIC, BIC, HQIC —")
            start_overall = time.time()

            var_model_for_order = VAR(train_pca)
            ic = var_model_for_order.select_order(maxlags=maxlags)

            criteria = ['aic', 'bic', 'hqic']
            best_results = {}
            lags_to_test = []
            for crit in criteria:
                best_lag = getattr(ic, crit)
                print(f"\nCriterion={crit.upper()}: selected lag p = {best_lag}")
                lags_to_test.append(best_lag)
            var_model = VAR(train_norm)

            for best_lag in np.unique(np.array(lags_to_test)):
                t0 = time.time()
                res_train = run_var(train_norm, train_norm, lag_order=best_lag)
                res_test = run_var(train_norm, test_norm, lag_order=best_lag)

                scores_train = res_train[W:]
                scores_test = res_test[W:]
                pot_th = compute_pot_threshold(scores_train, scores_test)
                preds_test = (scores_test >= pot_th).astype(int)

                prec, rec, f1 = compute_metrics(y_test_win, preds_test)
                elapsed = time.time() - t0

                print(f"  → Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
                print(f"  — VAR (lag={best_lag},  took {elapsed:.2f}s")

                save_results_csv(
                    args.results_dir,
                    args.dataset,
                    f"var_pca{n_keep}_{crit}_lag{best_lag}",
                    downsample=(args.downsample if args.downsample is not None else -1),
                    pot_thresh=pot_th,
                    prec=prec,
                    rec=rec,
                    f1=f1
                )

                best_results[crit] = (best_lag, prec, rec, f1)

            total_elapsed = time.time() - start_overall
            print(f"\nCompleted VAR+PCA experiments in {total_elapsed:.2f}s")
            for crit, (lag, prec, rec, f1) in best_results.items():
                print(f"  {crit.upper()}: lag={lag}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        elif baseline == "arma":
            import numpy as np
            from sklearn.decomposition import PCA

            print(f"— ARMA (p={args.arma_p}) —")

            if args.arma_topk is not None:
                pca_feat = PCA(n_components=min(train_norm.shape[1], train_norm.shape[0] - 1),
                               svd_solver='full')
                pca_feat.fit(train_norm)
                loadings = np.abs(pca_feat.components_[0])
                topk = args.arma_topk
                top_idxs = np.argsort(loadings)[-topk:]
                print(f"[PCA→ARMA] folosim top-{topk} canale: {top_idxs.tolist()}")
                arma_train = train_norm[:, top_idxs]
                arma_test = test_norm[:, top_idxs]
            else:
                arma_train = train_norm
                arma_test = test_norm

            start_time = time.time()
            r_train = run_arma(arma_train, arma_train, p=args.arma_p)
            r_test = run_arma(arma_train, arma_test, p=args.arma_p)

            scores_train = r_train[W:]
            scores_test = r_test[W:]
            pot_th = compute_pot_threshold(scores_train, scores_test)
            preds_test = (scores_test >= pot_th).astype(int)
            prec, rec, f1 = compute_metrics(y_test_win, preds_test)
            elapsed = time.time() - start_time

            print(f"    → Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            print(f"    — ARMA (p={args.arma_p}) took {elapsed:.2f}s")

            feature_tag = f"top{args.arma_topk}" if args.arma_topk is not None else "all"
            save_results_csv(
                args.results_dir,
                args.dataset,
                f"arma_p{args.arma_p}_{feature_tag}",
                downsample=(args.downsample if args.downsample is not None else -1),
                pot_thresh=pot_th,
                prec=prec,
                rec=rec,
                f1=f1
            )
