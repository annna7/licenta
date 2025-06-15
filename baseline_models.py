import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.kernel_approximation import Nystroem
from scipy.stats import genpareto
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_pot_threshold(
    init_scores: np.ndarray,
    test_scores: np.ndarray,
    q: float = 1e-5,
    init_level: float = 0.02,
    max_iters: int = 50
) -> float:
    level = init_level
    N = len(init_scores)
    success = False

    for _ in range(max_iters):
        u = np.quantile(init_scores, level)             # initial threshold “u”
        excesses = init_scores[init_scores > u] - u
        if len(excesses) < 5:
            # not enough tail points → lower u a bit
            level *= 0.9
            continue

        try:
            # try to fit GPD to excesses
            c, loc, scale = genpareto.fit(excesses, floc=0)
            p_u = len(excesses) / N

            if c != 0:
                thresh = u + (scale / c) * ((p_u / q) ** c - 1)
            else:
                # special case ξ = 0, exponential tail
                thresh = u - scale * np.log(q / p_u)

            pot_threshold = float(thresh)
            success = True
            break
        except Exception:
            level *= 0.9

    if not success:
        # fallback: simple percentile on test_scores
        fallback_pct = init_level * 100
        pot_threshold = np.percentile(test_scores, fallback_pct)
        print(f"[POT] GPD fitting failed – using {fallback_pct:.2f}th percentile = {pot_threshold:.4f}")

    return pot_threshold


def run_ocsvm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    kernel: str = 'rbf',
    subsample: int = None,
    nystroem_components: int = None,
    tophat_radius: float = None
) -> np.ndarray:
    if nystroem_components is not None:
        feature_map = Nystroem(kernel=kernel, n_components=nystroem_components)
        X_train = feature_map.fit_transform(X_train)
        X_test = feature_map.transform(X_test)
        kernel = 'linear'

    if (subsample is not None) and (subsample < X_train.shape[0]):
        idx = np.random.choice(X_train.shape[0], size=subsample, replace=False)
        X_train = X_train[idx]

    if kernel == 'tophat':
        if tophat_radius is None:
            sample_idx = np.random.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
            D = euclidean_distances(X_train[sample_idx], X_train[sample_idx])
            offdiag = D[np.triu_indices_from(D, k=1)]
            tophat_radius = np.percentile(offdiag, 95)
            print(f"[Tophat] estimated radius = {tophat_radius:.4f}")

        D_train = euclidean_distances(X_train, X_train)
        G_train = (D_train < tophat_radius).astype(np.float32)
        ocsvm = OneClassSVM(kernel='precomputed', gamma='auto')
        ocsvm.fit(G_train)

        D_test = euclidean_distances(X_test, X_train)
        G_test = (D_test < tophat_radius).astype(np.float32)
        scores = -ocsvm.decision_function(G_test)
        return scores

    ocsvm = OneClassSVM(kernel=kernel, gamma='auto')
    ocsvm.fit(X_train)
    scores = -ocsvm.decision_function(X_test)
    return scores


def run_iso_forest(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> np.ndarray:
    clf = IsolationForest(random_state=42, n_estimators=150)
    clf.fit(X_train)
    scores = -clf.decision_function(X_test)  # higher = more anomalous
    return scores


def run_lof(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 20
) -> np.ndarray:
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(X_train)
    scores = -clf.decision_function(X_test)
    return scores


def run_var(
    train_data: np.ndarray,
    test_data: np.ndarray,
    lag_order: int = 5
) -> np.ndarray:
    model = VAR(train_data)
    res = model.fit(lag_order)
    forecast = res.forecast(train_data[-lag_order:], steps=len(test_data))
    residuals = np.linalg.norm(test_data - forecast, axis=1)
    return residuals


def run_arma(
    train_data: np.ndarray,
    test_data: np.ndarray,
    p: int = 5
) -> np.ndarray:
    T_test, D = test_data.shape
    resid_test = np.zeros((T_test, D), dtype=np.float32)

    for i in range(D):
        # Fit AR(p) on each channel
        series_i = train_data[:, i]
        model_i = SARIMAX(
            series_i,
            order=(p, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res_i = model_i.fit(disp=False)
        forecast_i = res_i.get_forecast(steps=T_test).predicted_mean
        resid_test[:, i] = test_data[:, i] - forecast_i

    # collapse to single score sequence
    return np.linalg.norm(resid_test, axis=1)


def windowize(
    data: np.ndarray,
    window_size: int
) -> np.ndarray:
    T, C = data.shape
    if T <= window_size:
        return np.zeros((0, window_size * C))  # no windows if T <= window_size
    windows = []
    for i in range(T - window_size):
        w = data[i : i + window_size].reshape(-1)
        windows.append(w)
    return np.stack(windows, axis=0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    eps = 1e-10
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return prec, rec, f1
