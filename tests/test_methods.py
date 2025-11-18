import pytest
import torch
import networkx as nx
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Skip this test module entirely if RDKit is not available
pytest.importorskip("rdkit", reason="requires RDKit")
from rdkit import Chem

from molgraphx.methods import AtomsExplainer

def test_explainer_small_molecule_short_circuit():
    # predictor returns a constant scalar
    def predictor(mols):
        return torch.tensor([10.0 for _ in mols])

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
    graph = explainer._build_coalition_graph(mol)
    assert graph.molecules[0].GetNumAtoms() == mol.GetNumAtoms()
    assert len(graph.edge_atom_counts) == len(graph.parent_ids) == len(graph.child_ids)
    assert len(graph.edge_atom_indices) == sum(graph.edge_atom_counts)
    assert len(graph.edge_atom_edge_ids) == len(graph.edge_atom_indices)
    assert max(graph.edge_atom_edge_ids, default=-1) < len(graph.parent_ids)

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(len(graph.molecules)))
    nx_graph.add_edges_from(zip(graph.parent_ids, graph.child_ids))
    assert nx.is_directed_acyclic_graph(nx_graph)
