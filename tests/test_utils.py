import pytest
import torch

from src import utils as U


def test_gnn_gr_wrapper_picks_target_column():
    def model(batch):
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    f = U.GnnNetsGR2valueFunc(model, target=1)
    out = f(None)
    assert out == [2.0, 4.0]


def test_gnn_gc_wrapper_softmax_and_pick():
    def model(batch):
        return torch.tensor([[0.0, 0.0], [0.0, 1.0]])

    f = U.GnnNetsGC2valueFunc(model, target=1)
    out = f(None)
    # row0: [0.5, 0.5] -> pick 0.5; row1 softmax -> ~[0.2689, 0.7311]
    assert pytest.approx(out[0], rel=1e-6) == 0.5
    assert 0.7 < out[1] < 0.74


@pytest.mark.skipif(
    pytest.importorskip("rdkit", reason="requires RDKit") is None, reason="rdkit missing"
)
@pytest.mark.skipif(
    pytest.importorskip(
        "torch_geometric", reason="requires torch_geometric"
    )
    is None,
    reason="torch_geometric missing",
)
def test_get_scores_short_circuit_path(monkeypatch):
    from rdkit import Chem
    from torch_geometric.data import Data

    # Dummy featurizer returning minimal Data objects
    def featurizer(m):
        return Data()

    class DummyModel(torch.nn.Module):
        def forward(self, batch):  # batch is ignored
            return torch.zeros((1, 2))  # logits -> softmax -> [0.5, 0.5]

    mol = Chem.MolFromSmiles("CC")

    scores = U.get_scores(
        mol,
        featurizer=featurizer,
        explainable_model=DummyModel(),
        target=0,
        explainer_kwargs={"min_atoms": mol.GetNumAtoms()},
        mode="classification",
    )
    # Expect equal distribution: 0.5 / 2 per atom
    assert len(scores) == mol.GetNumAtoms()
    assert all(pytest.approx(s, rel=1e-6) == 0.25 for s in scores)

