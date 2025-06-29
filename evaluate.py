import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from util.data import *

# This implementation is based off the one provided in the original GDN paper:
# https://github.com/d-ailin/GDN
def aggregate_feature_errors(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        val_re_list = np_val_result[:2, :, i]

        scores = standardize_residuals(test_re_list, val_re_list)
        normal_dist = standardize_residuals(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def max_over_features(test_result, val_result):
    full_scores, all_normals = aggregate_feature_errors(test_result, val_result, return_normal_scores=True)

    return np.max(full_scores, axis=0)


def standardize_residuals(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)
    total_topk_err_scores = []

    for i, indexs in enumerate(topk_indices):
        sum_score = sum(
            score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])))

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def evaluate_threshold_performance(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    threshold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, threshold


def tune_threshold_for_best_f1(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    threshold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, threshold
