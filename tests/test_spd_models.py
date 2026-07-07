"""Tests for models/spd_models.py — SPDNet, SPDDecoder, ProtoSPDNet, MultiBandSPDNet."""
import pytest
import torch
import numpy as np

from features.spd_covariance import compute_covariance_scm
from models.spd_models import (
    BiMap,
    ReEig,
    LogEig,
    SPDNetModel,
    SPDDecoder,
    ProtoSPDNet,
    MultiBandSPDNet,
    create_spdnet,
    proto_loss,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def cov_input():
    """SPD covariance matrices: (B, C, C)."""
    rng = np.random.RandomState(42)
    X = rng.randn(8, 8, 250).astype(np.float32)
    C = compute_covariance_scm(X)
    return torch.from_numpy(C)


@pytest.fixture
def small_cov_input():
    """Larger batch of SPD matrices."""
    rng = np.random.RandomState(42)
    X = rng.randn(16, 8, 250).astype(np.float32)
    C = compute_covariance_scm(X)
    return torch.from_numpy(C)


# ── BiMap ────────────────────────────────────────────────────────────────────

class TestBiMap:
    def test_output_shape(self):
        layer = BiMap(8, 6)
        C = torch.randn(4, 8, 8)
        C = C @ C.transpose(-1, -2) + torch.eye(8) * 1e-3  # make SPD
        out = layer(C)
        assert out.shape == (4, 6, 6)

    def test_output_symmetric(self):
        layer = BiMap(8, 6)
        C = torch.randn(3, 8, 8)
        C = C @ C.transpose(-1, -2) + torch.eye(8) * 1e-3
        out = layer(C)
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)


# ── ReEig ────────────────────────────────────────────────────────────────────

class TestReEig:
    def test_output_shape(self):
        layer = ReEig()
        C = torch.randn(3, 6, 6)
        C = C @ C.transpose(-1, -2) + torch.eye(6)
        out = layer(C)
        assert out.shape == (3, 6, 6)

    def test_stays_symmetric(self):
        layer = ReEig()
        C = torch.randn(3, 6, 6)
        C = C @ C.transpose(-1, -2) + torch.eye(6)
        out = layer(C)
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)

    def test_stays_spd(self):
        """After ReEig, eigenvalues should be >= eps."""
        layer = ReEig(eps=0.01)
        C = torch.randn(3, 4, 4)
        C = C @ C.transpose(-1, -2) + torch.eye(4)
        out = layer(C)
        for i in range(3):
            eigvals = torch.linalg.eigvalsh(out[i])
            assert (eigvals >= 0.01 - 1e-4).all(), f"eigvals: {eigvals}"


# ── LogEig ───────────────────────────────────────────────────────────────────

class TestLogEig:
    def test_output_shape(self):
        layer = LogEig()
        C = torch.randn(3, 4, 4)
        C = C @ C.transpose(-1, -2) + torch.eye(4)
        out = layer(C)
        assert out.shape == (3, 4, 4)

    def test_symmetric(self):
        layer = LogEig()
        C = torch.randn(3, 4, 4)
        C = C @ C.transpose(-1, -2) + torch.eye(4)
        out = layer(C)
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)


# ── SPDNetModel ──────────────────────────────────────────────────────────────

class TestSPDNetModel:
    def test_default_init(self):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        assert m.n_classes == 2
        assert m.bimap_dims == [8, 6, 4]
        assert m.feat_dim == 4 * 5 // 2  # 10

    def test_output_shape(self, cov_input):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        with torch.no_grad():
            out = m(cov_input)
        assert out.shape == (8, 2)

    def test_single_sample(self):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        rng = np.random.RandomState(1)
        C = torch.from_numpy(compute_covariance_scm(rng.randn(1, 8, 250).astype(np.float32)))
        with torch.no_grad():
            out = m(C)
        assert out.shape == (1, 2)

    def test_deterministic(self, cov_input):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        with torch.no_grad():
            o1 = m(cov_input)
            o2 = m(cov_input)
        torch.testing.assert_close(o1, o2)

    def test_different_inputs(self):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        rng = np.random.RandomState(1)
        C1 = torch.from_numpy(compute_covariance_scm(rng.randn(4, 8, 250).astype(np.float32)))
        C2 = torch.from_numpy(compute_covariance_scm(rng.randn(4, 8, 250).astype(np.float32)))
        with torch.no_grad():
            o1 = m(C1)
            o2 = m(C2)
        assert not torch.allclose(o1, o2)

    def test_no_nan(self, cov_input):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        with torch.no_grad():
            out = m(cov_input)
        assert not torch.isnan(out).any()

    def test_3_class(self):
        m = SPDNetModel(n_classes=3, bimap_dims=[8, 6, 4])
        rng = np.random.RandomState(1)
        C = torch.from_numpy(compute_covariance_scm(rng.randn(4, 8, 250).astype(np.float32)))
        m.eval()
        with torch.no_grad():
            out = m(C)
        assert out.shape == (4, 3)

    def test_default_dims(self):
        m = SPDNetModel(n_classes=2)
        assert m.bimap_dims == [8, 6, 4]

    def test_gradient_flows(self, cov_input):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.train()
        out = m(cov_input)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad: {name}"

    def test_save_load_roundtrip(self, cov_input, tmp_path):
        m = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m.eval()
        with torch.no_grad():
            out_before = m(cov_input)

        path = tmp_path / "spdnet.pt"
        torch.save(m.state_dict(), path)

        m2 = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
        m2.eval()
        m2.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            out_after = m2(cov_input)
        torch.testing.assert_close(out_before, out_after)


# ── create_spdnet ────────────────────────────────────────────────────────────

class TestCreateSPDNet:
    def test_defaults(self):
        m = create_spdnet(n_channels=8, n_classes=2)
        assert isinstance(m, SPDNetModel)
        assert m.n_classes == 2

    def test_custom(self):
        m = create_spdnet(n_channels=8, n_classes=3, bimap_dims=[8, 5, 3])
        assert m.bimap_dims == [8, 5, 3]

    def test_auto_dims(self, cov_input):
        m = create_spdnet(n_channels=8, n_classes=2)  # auto dims
        m.eval()
        with torch.no_grad():
            out = m(cov_input)
        assert out.shape == (8, 2)


# ── SPDDecoder ───────────────────────────────────────────────────────────────

class TestSPDDecoder:
    def test_output_shape(self):
        decoder = SPDDecoder(feat_dim=10, hidden_dim=64)
        z = torch.randn(4, 10)
        recon = decoder(z)
        assert recon.shape == (4, 10)

    def test_produces_finite(self):
        decoder = SPDDecoder(feat_dim=10)
        z = torch.randn(4, 10)
        recon = decoder(z)
        assert torch.isfinite(recon).all()


# ── ProtoSPDNet ──────────────────────────────────────────────────────────────

class TestProtoSPDNet:
    def test_output_shape(self, cov_input):
        m = ProtoSPDNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(cov_input)
        assert out.shape == (8, 2)

    def test_return_features(self, cov_input):
        m = ProtoSPDNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            logits, feats = m(cov_input, return_features=True)
        assert logits.shape == (8, 2)
        assert feats.ndim == 2
        assert feats.shape[0] == 8

    def test_prototypes_exist(self):
        m = ProtoSPDNet(n_channels=8, n_classes=3)
        assert m.prototypes.shape == (3, m.feat_dim)

    def test_gradient_flows(self, cov_input):
        m = ProtoSPDNet(n_channels=8, n_classes=2)
        m.train()
        labels = torch.randint(0, 2, (8,))
        logits, feats = m(cov_input, return_features=True)
        ce_loss = torch.nn.functional.cross_entropy(logits, labels)
        p_loss = proto_loss(feats, labels, m.prototypes, temperature=0.1)
        loss = ce_loss + 0.1 * p_loss
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad: {name}"


# ── proto_loss ───────────────────────────────────────────────────────────────

class TestProtoLoss:
    def test_scalar_output(self):
        feats = torch.randn(16, 10)
        labels = torch.randint(0, 3, (16,))
        prots = torch.randn(3, 10)
        loss = proto_loss(feats, labels, prots)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_decreases_with_gt_aligned(self):
        """Loss should be low when features ≈ correct prototype."""
        prots = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        feats = prots[[0, 1, 2, 0, 1, 2]] + torch.randn(6, 2) * 0.001
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = proto_loss(feats, labels, prots, temperature=0.01)
        assert loss.item() < 0.5


# ── MultiBandSPDNet ──────────────────────────────────────────────────────────

class TestMultiBandSPDNet:
    @pytest.fixture
    def band_covs(self):
        rng = np.random.RandomState(42)
        X = rng.randn(4, 8, 250).astype(np.float32)
        return {
            "mu": torch.from_numpy(compute_covariance_scm(X)),
            "beta": torch.from_numpy(compute_covariance_scm(X + 0.1)),
        }

    def test_shared_branches(self, band_covs):
        m = MultiBandSPDNet(n_bands=2, n_channels=8, n_classes=2,
                            bimap_dims=[8, 6], share_branches=True)
        m.eval()
        with torch.no_grad():
            out = m(band_covs)
        assert out.shape == (4, 2)

    def test_separate_branches(self, band_covs):
        m = MultiBandSPDNet(n_bands=2, n_channels=8, n_classes=2,
                            bimap_dims=[8, 6], share_branches=False)
        m.eval()
        with torch.no_grad():
            out = m(band_covs)
        assert out.shape == (4, 2)

    def test_no_nan(self, band_covs):
        m = MultiBandSPDNet(n_bands=2, n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(band_covs)
        assert not torch.isnan(out).any()

    def test_gradient_shared(self, band_covs):
        m = MultiBandSPDNet(n_bands=2, n_channels=8, n_classes=2,
                            share_branches=True)
        m.train()
        out = m(band_covs)
        loss = out.sum()
        loss.backward()
        # Shared mode: branch.classifier is unused; only m.classifier receives grad
        skip = {"branch.classifier.weight", "branch.classifier.bias"}
        for name, p in m.named_parameters():
            if name in skip:
                continue
            assert p.grad is not None, f"No grad: {name}"

    def test_gradient_separate(self, band_covs):
        m = MultiBandSPDNet(n_bands=2, n_channels=8, n_classes=2,
                            share_branches=False)
        m.train()
        out = m(band_covs)
        loss = out.sum()
        loss.backward()
        # Separate mode: branches[i].classifier is unused; only m.classifier receives grad
        skip = {f"branches.{i}.classifier.weight" for i in range(2)} | \
               {f"branches.{i}.classifier.bias" for i in range(2)}
        for name, p in m.named_parameters():
            if name in skip:
                continue
            assert p.grad is not None, f"No grad: {name}"
