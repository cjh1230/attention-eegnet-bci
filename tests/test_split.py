"""Tests for training/split.py — subject-wise data splitting utilities."""
import numpy as np
import pytest

from training.split import subject_wise_split, loso_splits, kfold_subject_split


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def subjects():
    """10 synthetic subjects with X, y data."""
    rng = np.random.RandomState(42)
    subjs = []
    for i in range(10):
        subjs.append({
            "id": f"S{i:02d}",
            "X": rng.randn(18, 8, 250).astype(np.float32),
            "y": rng.randint(0, 2, 18).astype(np.int64),
        })
    return subjs


@pytest.fixture
def subjects_3():
    """3 subjects for edge-case testing."""
    rng = np.random.RandomState(1)
    return [
        {"id": f"S{i:02d}", "X": rng.randn(6, 8, 250).astype(np.float32),
         "y": rng.randint(0, 2, 6).astype(np.int64)}
        for i in range(3)
    ]


# ── subject_wise_split ──────────────────────────────────────────────────────

class TestSubjectWiseSplit:
    def test_returns_train_test(self, subjects):
        train, test = subject_wise_split(subjects, test_size=0.25)
        assert len(train) + len(test) == len(subjects)
        assert len(test) >= 1
        assert len(train) >= 1

    def test_no_overlap(self, subjects):
        train, test = subject_wise_split(subjects, test_size=0.3)
        train_ids = {s["id"] for s in train}
        test_ids = {s["id"] for s in test}
        assert train_ids & test_ids == set()

    def test_all_ids_present(self, subjects):
        train, test = subject_wise_split(subjects, test_size=0.2)
        all_ids = {s["id"] for s in train} | {s["id"] for s in test}
        assert all_ids == {s["id"] for s in subjects}

    def test_test_size_1_0_uses_all_train(self, subjects):
        train, test = subject_wise_split(subjects, test_size=1.0)
        assert len(test) == len(subjects)
        assert len(train) == 0

    def test_min_test_size_rounds_up(self, subjects):
        """test_size very small still gives at least 1 test subject."""
        train, test = subject_wise_split(subjects, test_size=0.01)
        assert len(test) == 1
        assert len(train) == len(subjects) - 1

    def test_reproducible_with_seed(self, subjects):
        train1, test1 = subject_wise_split(subjects, test_size=0.3, random_state=42)
        train2, test2 = subject_wise_split(subjects, test_size=0.3, random_state=42)
        ids1 = [s["id"] for s in test1]
        ids2 = [s["id"] for s in test2]
        assert ids1 == ids2

    def test_different_seeds_differ(self, subjects):
        train1, test1 = subject_wise_split(subjects, test_size=0.3, random_state=42)
        train2, test2 = subject_wise_split(subjects, test_size=0.3, random_state=123)
        ids1 = [s["id"] for s in test1]
        ids2 = [s["id"] for s in test2]
        # Not guaranteed to differ but extremely likely with 10 subjects
        assert ids1 != ids2 or len(subjects) <= 1

    def test_preserves_data_integrity(self, subjects):
        train, test = subject_wise_split(subjects, test_size=0.3)
        # Check that subject data is not mutated
        for s in train + test:
            assert s["X"].shape == (18, 8, 250)
            assert s["y"].shape == (18,)


# ── loso_splits ─────────────────────────────────────────────────────────────

class TestLOSOSplits:
    def test_n_folds_equals_n_subjects(self, subjects):
        folds = loso_splits(subjects)
        assert len(folds) == len(subjects)

    def test_each_subject_is_test_exactly_once(self, subjects):
        folds = loso_splits(subjects)
        test_ids = []
        for train_subjs, test_subj in folds:
            test_ids.append(test_subj["id"])
            # Train does not contain test subject
            train_ids = {s["id"] for s in train_subjs}
            assert test_subj["id"] not in train_ids
        # Each subject appears as test exactly once
        assert sorted(test_ids) == sorted(s["id"] for s in subjects)

    def test_train_size_n_minus_1(self, subjects):
        folds = loso_splits(subjects)
        for train_subjs, _ in folds:
            assert len(train_subjs) == len(subjects) - 1

    def test_no_data_mutation(self, subjects):
        folds = loso_splits(subjects)
        for train_subjs, test_subj in folds:
            assert test_subj["X"].shape[0] > 0
            for s in train_subjs:
                assert s["X"].shape[0] > 0

    def test_single_subject(self):
        """With 1 subject, train is empty."""
        single = [{"id": "S00", "X": np.ones((6, 8, 250)), "y": np.zeros(6)}]
        folds = loso_splits(single)
        assert len(folds) == 1
        train, test = folds[0]
        assert len(train) == 0
        assert test["id"] == "S00"

    def test_3_subjects(self, subjects_3):
        folds = loso_splits(subjects_3)
        assert len(folds) == 3

    def test_consistent_order(self, subjects):
        folds1 = loso_splits(subjects)
        folds2 = loso_splits(subjects)
        for (t1, s1), (t2, s2) in zip(folds1, folds2):
            assert s1["id"] == s2["id"]


# ── kfold_subject_split ─────────────────────────────────────────────────────

class TestKfoldSubjectSplit:
    def test_n_folds(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=5)
        assert len(folds) == 5

    def test_no_overlap_per_fold(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=5)
        for train, test in folds:
            train_ids = {s["id"] for s in train}
            test_ids = {s["id"] for s in test}
            assert train_ids & test_ids == set()

    def test_all_subjects_covered(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=5)
        all_test_ids = set()
        for _, test in folds:
            all_test_ids |= {s["id"] for s in test}
        assert all_test_ids == {s["id"] for s in subjects}

    def test_each_subject_in_exactly_one_test_fold(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=5)
        test_counts = {}
        for _, test in folds:
            for s in test:
                test_counts[s["id"]] = test_counts.get(s["id"], 0) + 1
        for count in test_counts.values():
            assert count == 1

    def test_fold_sizes_roughly_equal(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=5)
        sizes = [len(test) for _, test in folds]
        assert max(sizes) - min(sizes) <= 1

    def test_n_folds_equals_n_subjects(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=10)
        assert len(folds) == 10
        for _, test in folds:
            assert len(test) == 1  # each fold has exactly 1 test subject

    def test_n_folds_2(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=2)
        assert len(folds) == 2
        total_test = sum(len(test) for _, test in folds)
        assert total_test == len(subjects)

    def test_invalid_n_folds_too_small(self, subjects):
        with pytest.raises(ValueError, match="n_folds must be"):
            kfold_subject_split(subjects, n_folds=1)

    def test_invalid_n_folds_too_large(self, subjects):
        with pytest.raises(ValueError, match="n_folds must be"):
            kfold_subject_split(subjects, n_folds=len(subjects) + 1)

    def test_reproducible_with_seed(self, subjects):
        folds1 = kfold_subject_split(subjects, n_folds=5, random_state=42)
        folds2 = kfold_subject_split(subjects, n_folds=5, random_state=42)
        ids1 = sorted([s["id"] for _, test in folds1 for s in test])
        ids2 = sorted([s["id"] for _, test in folds2 for s in test])
        # The fold membership should be the same
        for f1, f2 in zip(folds1, folds2):
            assert sorted(s["id"] for s in f1[1]) == sorted(s["id"] for s in f2[1])

    def test_preserves_data_integrity(self, subjects):
        folds = kfold_subject_split(subjects, n_folds=3)
        for train, test in folds:
            for s in train + test:
                assert s["X"].shape == (18, 8, 250)
                assert s["y"].shape == (18,)
