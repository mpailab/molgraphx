import pytest
import networkx as nx
from rdkit import Chem

from src.methods import AtomsExplainer


def test_explainer_small_molecule_short_circuit():
    # predictor returns a constant scalar
    def predictor(mols):
        return [10.0 for _ in mols]

    mol = Chem.MolFromSmiles("CC")
    explainer = AtomsExplainer(predictor, min_atoms=mol.GetNumAtoms())
    scores = explainer(mol)
    # equal split across atoms: 10 / 2 each
    assert len(scores) == mol.GetNumAtoms()
    assert all(abs(s - 5.0) < 1e-6 for s in scores)


def test_build_coalition_graph_is_dag_and_has_root():
    def predictor(mols):
        # simple monotone predictor by atom count
        return [float(m.GetNumAtoms()) for m in mols]

    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    explainer = AtomsExplainer(predictor, min_atoms=3)
    g = explainer._build_coalition_graph(mol)
    assert isinstance(g, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(g)
    # root coalition is all atom indices
    root = frozenset(range(mol.GetNumAtoms()))
    assert root in g.nodes

