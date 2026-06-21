"""
Artifact removal helpers (ICA-based).
"""
import mne


def apply_ica(
    raw: mne.io.Raw,
    n_components: int = 15,
    method: str = "fastica",
    random_state: int = 42,
) -> mne.io.Raw:
    """Fit ICA and remove artifact components automatically."""
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
    )
    ica.fit(raw)
    # Auto-detect EOG artifacts (silently skip if no EOG channels)
    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=None, verbose=False)
        ica.exclude = eog_indices
    except RuntimeError:
        pass  # no EOG channels available
    raw_clean = ica.apply(raw.copy())
    return raw_clean
