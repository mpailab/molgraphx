import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Skip this test module entirely if RDKit is not available
pytest.importorskip("rdkit", reason="requires RDKit")
from rdkit import Chem

from molgraphx.molecule import symmetry_classes, submolecule


def test_find_mol_sym_atoms_equivalent_in_ethane():
    mol = Chem.MolFromSmiles("CC")
    sym = symmetry_classes(mol)
    # carbons 0 and 1 are equivalent in ethane
    assert len(sym) == 1
    assert sym[0] == set({0, 1})


def test_find_mol_sym_atoms_non_equivalent_in_etho():
    mol = Chem.MolFromSmiles("CCO")
    sym = symmetry_classes(mol)
    # terminal carbon and oxygen are not equivalent
    assert 2 not in sym[0]
    assert 0 not in sym[2]


def test_submolecule_connected_fragment_and_mapping():
    mol = Chem.MolFromSmiles("CCO")
    atoms = frozenset({0, 1})
    atom_maps: list[int] = []
    sub = submolecule(mol, atoms, atom_maps)
    assert sub.GetNumAtoms() == 2
    # mapping from sub indices to original indices should be 0 and 1 in some order
    assert set(atom_maps) == {0, 1}


def test_submolecule_aromatic_subset():
    # three adjacent atoms in benzene should form a 3-atom chain
    mol = Chem.MolFromSmiles("c1ccccc1")
    atoms = frozenset({0, 1, 2})
    atom_maps: list[int] = []
    sub = submolecule(mol, atoms, atom_maps)
    assert sub.GetNumAtoms() == 3
    assert sub.GetNumBonds() == 2


def test_submolecule_no_internal_bonds_returns_empty():
    mol = Chem.MolFromSmiles("CCO")
    atoms = frozenset({0, 2})  # not directly bonded in this molecule
    sub = submolecule(mol, atoms)
    assert sub.GetNumAtoms() == 0


def test_submolecule_drop_disconnected_result():
    # Two separate ethane fragments
    mol = Chem.MolFromSmiles("CC.CC")
    atoms = frozenset(range(mol.GetNumAtoms()))
    # disconnected result should drop and return empty
    sub_drop = submolecule(mol, atoms)
    assert sub_drop.GetNumAtoms() == 0
    # previous behavior with connected=False is no longer available


def test_submolecule_subset_disconnected_in_same_molecule():
    # In ethanol, atoms {0,2} are disconnected in the induced subgraph
    mol = Chem.MolFromSmiles("CCO")
    atoms = frozenset({0, 2})
    empty = submolecule(mol, atoms)
    assert empty.GetNumAtoms() == 0
