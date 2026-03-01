import math
import numpy as np


def estimate_mv_error(y_gt, preds, k=3, condition=None):
    """
    Estimate the majority voting classifier error L(h_3^{MV}).
    Parameters:
        y_gt: Ground truth labels, shape (N,)
        preds: Prediction matrix, shape (N, 3) where each column corresponds to the predictions from one classifier.
        k: Parameter k (default is k=3)
        condition: If specified, compute the error rate only on samples where y_gt == condition (e.g., 1 for FN rate)
    """
    if condition is not None:
        mask = y_gt == condition
        y_gt = y_gt[mask]
        preds = preds[mask, :]
        
    N = y_gt.shape[0]
    
    # 1. Compute the error indicator for each classifier and each sample (1 if error, 0 if correct)
    err = (preds != y_gt[:, np.newaxis]).astype(float)  # shape (N, 3)
    L_bar = np.mean(err)
    
    # 2. Compute the average disagreement D_bar among classifiers
    d01 = (preds[:, 0] != preds[:, 1]).astype(float)
    d02 = (preds[:, 0] != preds[:, 2]).astype(float)
    d12 = (preds[:, 1] != preds[:, 2]).astype(float)
    D_bar = np.mean(np.concatenate([d01, d02, d12]))
    
    # 3. Compute epsilon_3
    eps_numerators = []
    for i in range(N):
        eps_num = 0.0
        # Iterate over all classifier pairs (j, jp)
        for j in range(3):
            for jp in range(3):
                # Indicator for the disagreement between classifier j and classifier jp for sample i
                indicator_diff = 1.0 if preds[i, j] != preds[i, jp] else 0.0
                # eps_num += err[i, j] * err[i, jp]
                # eps_num += err[i, j] * err[i, jp] * indicator_diff
                eps_num += 0.5 * (err[i, j] + err[i, jp]) * indicator_diff
        eps_numerators.append(eps_num / 9.0)
    eps_avg = np.mean(eps_numerators)
    epsilon_3 = eps_avg / (2 * L_bar) if L_bar > 0 else 0.0
    # epsilon_3 = (3 - 2) / (2 * (3 - 1))

    # 4. Compute eta3
    W = np.mean(err, axis=1)      # Compute W(x) for each sample
    numerator = np.mean(W > 0.5)  # Proportion of samples with W(x) > 0.5
    denominator = np.mean(W**2)   # Average of W(x)^2 over all samples
    eta3 = numerator / denominator if denominator > 0 else 0.0
    
    # 5. Estimate L(h_3^{MV})
    bracket = L_bar + (3 * (k - 1)) / (2 * k) * (epsilon_3 * L_bar - 0.5 * D_bar)
    L_mv_est = eta3 * bracket  
    
    return L_mv_est


def estimate_mv_error_fn(y_gt, preds, k=3):
    """
    Post-correction error rate = FN/N
    """
    fn_conditional = estimate_mv_error(y_gt, preds, k=k, condition=1)
    p_y1 = np.mean(y_gt == 1)
    post_correction_error = fn_conditional * p_y1
    return round(post_correction_error, 4)


if __name__ == '__main__':
    example_y_gt = np.array([0, 1, 1, 0, 1])
    example_preds = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1]
    ])
    
    for _k in range(3, 9):
        example_l_mv_est = estimate_mv_error(example_y_gt, example_preds, k=_k)
        print(f"when k={_k}, estimated majority voting error = {example_l_mv_est:.4f}")


def chernoff_bound(N, p_e):
    """
    Calculate Chernoff bound for probability P(X > N/2)
    Parameters:
        N (int): Number of classifiers (ensemble size)
        p_e (float): Error probability of individual classifier
    """
    if p_e <= 0.0 or p_e >= 1.0:
        return 0.0
    delta = (1 / (2 * p_e)) - 1
    exponent = -((delta ** 2) * N * p_e) / 2
    bound = math.exp(exponent)
    return bound
