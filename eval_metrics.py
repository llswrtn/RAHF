import numpy as np
from sklearn.metrics import auc

### Heatmaps ###

"""
we report MSE on all samples and on those with empty ground truth, respectively,
and report saliency heatmap evaluation metrics like NSS/KLD/AUC-Judd/SIM/CC [5]
for the samples with non-empty ground truth.

[5] Zoya Bylinskii, Tilke Judd, Aude Oliva, Antonio Torralba, and Fre ́do Durand.
    What do different evaluation metrics tell us about saliency models? IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018
"""
import torch
import numpy as np
#from sklearn.metrics import roc_auc_score
#from pysaliency import AUC, KLDivergence, NSS, Similarity, CorrelationCoefficient

# Normalized Scanpath Saliency (NSS)
def NSS(pred, gt):
    """
    pred: Predicted saliency map (2D array, normalized to [0, 1]).
    gt: Ground truth map (binary 2D array where 1 indicates fixation).

    from : Peters et al. 2005: Components of bottom-up gaze allocation in natural images
    Each salience map was linearly normalized to have zero mean and unit standard deviation.
    Next, the normalized salience values were extracted from each point corresponding to the
    fixation locations along a subject’s scanpath, and the mean of these values, or normalized
    scanpath salience (NSS), was taken as a measure of the correspondence between the salience
    map and scanpath. Due to the pre-normalization of the salience map, normalized scanpath
    salience values greater than zero suggest a greater correspondence than would be expected
    by chance between fixation locations and the salient points predicted by the model; a value
    of zero indicates no such correspondence, while values less than zero indicate an anti-
    correspondence between fixation locations and model-predicted salient points. Another
    benefit of the pre-normalization is that these measures could be compared across different
    subjects, image classes, and model variants; with such a data pool, statistical tests
    indicated whether the distribution of NSS values was different from the zero-mean
    distribution expected by chance.

    compare https://github.com/matthias-k/pysaliency/blob/dev/pysaliency/metrics.py
    """

    # normalize alternative: divide by 255, set all nonzero values in gt to 1 (other options??)
    gt[gt != 0] = 1
    #pred = normalize_to_range_0_1(pred)

    pred_mean = np.mean(pred)
    pred_std = np.std(pred)

    if pred_std == 0:
        return 0  # Avoid division by zero

    # normalize the pred map to have zero mean and std of one
    pred_normalized = (pred - pred_mean) / pred_std

    # Compute NSS
    # mean of each value corresponding to a fixation (here taking any point in gt heatmap greater 0)
    nss = np.mean(pred_normalized[gt == 1])
    return nss

# AUC Judd
def AUC_Judd(saliency_map, fixation_map):
    """
    Compute AUC-Judd score for saliency map evaluation.

    Parameters:
        saliency_map (numpy.ndarray): Predicted saliency map (grayscale values).
        fixation_map (numpy.ndarray): Binary map of fixation points (1 for fixations, 0 otherwise).

    Returns:
        float: AUC-Judd score.
    """

    def calculate_tpr_fpr(s_map, f_map, thresholds):
        """
        Calculate the True Positive Rate (TPR) and False Positive Rate (FPR) for given thresholds.

        Parameters:
        - saliency_map: 2D numpy array representing the saliency map.
        - fixation_map: 2D binary numpy array where 1 indicates fixation and 0 otherwise.
        - thresholds: List or array of thresholds to evaluate.

        Returns:
        - tpr: List of True Positive Rates for each threshold.
        - fpr: List of False Positive Rates for each threshold.
        """
        tpr = []

        fpr = []

        total_fixations = np.sum(f_map)  # Total number of fixations (positive samples)
        total_pixels = s_map.size  # Total number of pixels in the saliency map

        for thresh in thresholds:
            # Create a binary mask for the current threshold
            level_set = s_map >= thresh

            # Calculate true positives (TP)
            true_positives = np.sum(level_set * f_map)

            # Calculate false positives (FP)
            false_positives = np.sum(level_set) - true_positives

            # True Positive Rate (TPR)
            tpr.append(true_positives / total_fixations if total_fixations > 0 else 0)

            # False Positive Rate (FPR)
            fpr.append(false_positives / total_pixels if total_pixels > 0 else 0)

        return tpr, fpr

    # 1. convert gt map to binary fixation map
    fixation_map[fixation_map != 0] = 1

    # 2. compute thresholds

    saliencies_at_fixations = saliency_map * fixation_map # thresholds from saliency map values only where fixations exist (AUC_Judd, see AUC_per_image here:https://github.com/matthias-k/pysaliency/blob/dev/pysaliency/saliency_map_models.py#L242
    saliencies_at_fixations = saliencies_at_fixations[saliencies_at_fixations != 0]
    thresholds = np.sort(np.unique(saliencies_at_fixations))


    # Get TP rate and FP rate

    tpr, fpr = calculate_tpr_fpr(saliency_map, fixation_map, thresholds)

    # Sort FPR and TPR for the AUC calculation (optional, if needed)
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]

    # Compute the AUC with sklearn
    auc_value = auc(fpr, tpr)
    return auc_value




##### Custom Implementations, for reference only, not used in eval #####

def normalize_to_density(saliency_map):
    """
    Normalize a saliency map so that it sums to 1 (probability distribution).
    saliency_map: 2D array
    """
    total_sum = np.sum(saliency_map)

    if total_sum == 0:
        return np.zeros(saliency_map)  # Avoid division by zero

    return saliency_map / total_sum

def KLD_custom (pred, gt):
    """
    pred: Predicted saliency map (normalized to sum to 1).
    gt: Ground truth saliency map (normalized to sum to 1).

    following [5] cited in original paper, KL-Judd
    [5] Zoya Bylinskii, Tilke Judd, et al.
    What do different evaluation metrics tell us about saliency models?
    """
    reg_const = 2.2204e-16 #footnote 3 page 10 in [5]

    pred_norm = normalize_to_density(pred)
    gt_norm = normalize_to_density(gt)

    log_pred = np.log(pred_norm)
    log_gt = np.log(gt_norm)
    return (np.exp(log_gt) * np.log(reg_const + np.exp(log_gt) / (np.exp(log_pred) + reg_const))).sum()


# Similarity (SIM)
def SIM_custom (pred, gt):
    """
    pred: Predicted saliency map (normalize to sum to 1).
    gt: Ground truth saliency map (normalize to sum to 1).
    """
    pred = normalize_to_density(pred)
    gt = normalize_to_density(gt)
    return np.sum(np.minimum(pred, gt))

# Linear Correlation Coefficient (CC)


def CC_custom(pred, gt):
    """
    pred: Predicted saliency map (normalized to sum to 1).
    gt: Ground truth saliency map (normalized to sum to 1).
    """
    pred_mean = np.mean(pred)
    gt_mean = np.mean(gt)
    numerator = np.sum((pred - pred_mean) * (gt - gt_mean))
    denominator = np.sqrt(np.sum((pred - pred_mean) ** 2) * np.sum((gt - gt_mean) ** 2))
    return numerator / (denominator + 1e-8)

# not necessary: output heatmaps and getitem in Dataset class already range [0, 1]
'''
def normalize_to_range_0_1(saliency_map):
    """
    Normalize a saliency map to the range [0, 1].
    saliency_map: 2D array or tensor
    """
    min_val = torch.min(saliency_map)
    max_val = torch.max(saliency_map)

    if max_val - min_val == 0:
        return torch.zeros_like(saliency_map)  # Avoid division by zero

    return (saliency_map - min_val) / (max_val - min_val)
'''

