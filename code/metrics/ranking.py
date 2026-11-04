"""Do two experiments rank the methods in the same order? (Used to ask whether synthetic data can stand in for real data.)"""
from scipy.stats import kendalltau, spearmanr


def ranking_agreement(scores_a, scores_b):
    """Compare two lists of scores, one number per method in the same order, and return how alike the rankings are.

    Spearman and Kendall both run from -1 (opposite order) through 0 (unrelated) to +1 (identical order).
    Kendall is the more cautious of the two with only a few methods.
    """
    spearman = spearmanr(scores_a, scores_b).statistic
    kendall = kendalltau(scores_a, scores_b).statistic
    return {"spearman": float(spearman), "kendall": float(kendall)}
