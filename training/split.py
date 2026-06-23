"""
Subject-wise data splitting utilities for BCI evaluation.

BCI data is inherently nested: trials within subjects. Random trial-level
splitting leaks subject identity across train/test, inflating accuracy.
These functions enforce subject-level separation for credible evaluation.

Strategies:
  - subject_wise_split  — stratified hold-out by subject
  - loso_splits         — Leave-One-Subject-Out generator
  - kfold_subject_split — K-fold cross-validation at subject level
"""

from typing import Optional
import numpy as np


def subject_wise_split(
    subjects: list[dict],
    test_size: float = 0.25,
    random_state: Optional[int] = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split subjects into train / test sets.

    Parameters
    ----------
    subjects : list[dict]
        Each entry must have keys 'id', 'X', 'y'.
    test_size : float
        Fraction of subjects to hold out (0 < test_size < 1).
    random_state : int or None

    Returns
    -------
    train_subjs, test_subjs : tuple[list[dict], list[dict]]
    """
    rng = np.random.RandomState(random_state)
    n_test = max(1, int(len(subjects) * test_size))
    indices = rng.permutation(len(subjects))
    test_idx = set(indices[:n_test].tolist())
    train_subjs = [s for i, s in enumerate(subjects) if i not in test_idx]
    test_subjs = [s for i, s in enumerate(subjects) if i in test_idx]
    return train_subjs, test_subjs


def loso_splits(subjects: list[dict]) -> list[tuple[list[dict], dict]]:
    """
    Leave-One-Subject-Out generator.

    Yields all possible (train_subjects, test_subject) folds.
    For N subjects, produces N folds.

    Parameters
    ----------
    subjects : list[dict]
        Each entry must have keys 'id', 'X', 'y'.

    Returns
    -------
    folds : list[tuple[list[dict], dict]]
        Each element is (train_subjs_list, test_subj_dict).
    """
    folds = []
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s["id"] != test_subj["id"]]
        folds.append((train_subjs, test_subj))
    return folds


def kfold_subject_split(
    subjects: list[dict],
    n_folds: int = 5,
    random_state: Optional[int] = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """
    K-fold cross-validation at subject level.

    Divides subjects into *n_folds* groups. Each fold uses one group
    as test and the rest as train.

    Parameters
    ----------
    subjects : list[dict]
    n_folds : int
        Number of folds (2 ≤ n_folds ≤ len(subjects)).
    random_state : int or None

    Returns
    -------
    folds : list[tuple[list[dict], list[dict]]]
        Each element is (train_subjs, test_subjs).
    """
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(f"n_folds must be in [2, {len(subjects)}], got {n_folds}")

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(subjects))
    fold_size = len(subjects) // n_folds
    folds = []

    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else len(subjects)
        test_idx = set(indices[start:end].tolist())
        train_subjs = [s for i, s in enumerate(subjects) if i not in test_idx]
        test_subjs = [s for i, s in enumerate(subjects) if i in test_idx]
        folds.append((train_subjs, test_subjs))

    return folds
