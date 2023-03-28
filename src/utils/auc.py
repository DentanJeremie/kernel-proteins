def roc_auc_score(y_true, y_score):
    """
    Computes the AUC score given an array of true labels and an array of predicted probabilities or scores.

    Parameters:
    y_true (array-like): True labels of the binary classification problem.
    y_score (array-like): Predicted probabilities or scores for the positive class.

    Returns:
    float: The AUC score.
    """
    # Sort the true labels and scores in descending order of the scores.
    sorted_indices = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    y_true_sorted = [y_true[i] for i in sorted_indices]
    y_score_sorted = [y_score[i] for i in sorted_indices]

    # Compute the cumulative sums of the true positive and false positive rates.
    cum_tp = [0]
    cum_fp = [0]
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            cum_tp.append(cum_tp[-1] + 1)
            cum_fp.append(cum_fp[-1])
        else:
            cum_tp.append(cum_tp[-1])
            cum_fp.append(cum_fp[-1] + 1)

    # Compute the true positive rate and false positive rate for each threshold.
    tpr = [x / cum_tp[-1] for x in cum_tp]
    fpr = [x / cum_fp[-1] for x in cum_fp]

    # Compute the AUC score as the area under the ROC curve.
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0

    return auc
