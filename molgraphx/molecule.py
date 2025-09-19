# External imports
import networkx as nx
from rdkit.Chem import RWMol, SanitizeMol, GetMolFrags, PathToSubmol
from rdkit.Chem.rdmolops import SanitizeFlags
from rdkit.Chem.rdchem import Atom, Bond, BondType, Mol
from typing import FrozenSet, Dict, List


def get_atom_props(atom : Atom, ring_size : int = 1) -> tuple:
    """
    Get properties of atom

    Parameters
    ----------
    atom : Atom
        Atom
    ring_size : int
        The maximal ring size to look for

    Returns
    -------
    atom_props : tuple
        Tuple of atom's properties
    """
    return tuple([
        atom.GetSymbol(), 
        atom.GetDegree(), 
        atom.GetTotalDegree(), 
        atom.GetImplicitValence(), 
        atom.GetExplicitValence(), 
        atom.GetTotalValence(), 
        atom.GetNumImplicitHs(), 
        atom.GetNumExplicitHs(), 
        atom.GetTotalNumHs(), 
        atom.GetIsAromatic(),
        atom.IsInRing(), 
        *[ atom.IsInRingSize(i) for i in range(ring_size) ]
    ])


def get_bond_props(bond : Bond, atom : Atom, atoms_to_sym_cls : dict) -> tuple:
    """
    Get properties of bond

    Parameters
    ----------
    bond : Bond
        Bond
    atom : Atom
        Bond's atom
    atoms_to_sym_cls : dict { atom_idx (int) -> sym_class_idx (int) }
        Table of atoms symmetry classes

    Returns
    -------
    bond_props : tuple
        Tuple of bond's properties
    """
    return tuple([
        int(bond.GetBondDir()),
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated()),
        atoms_to_sym_cls[bond.GetOtherAtom(atom).GetIdx()]
    ])


def get_atom_bonds_props(atom : Atom, atoms_to_sym_cls : dict) -> tuple:
    """
    Get bonds properties for atom

    Parameters
    ----------
    atom : Atom
        Atom
    atoms_to_sym_cls : dict { atom_idx (int) -> sym_class_idx (int) }
        Table of atoms symmetry classes

    Returns
    -------
    bonds_props : tuple
        Sorted tuple of bonds properties for atom
    """
    bonds_props = [
        get_bond_props(bond, atom, atoms_to_sym_cls) for bond in atom.GetBonds()
    ]
    if len(bonds_props) > 1:
        # Sort bonds properties in alphabetical order
        for i in range(len(bonds_props[0])):
            bonds_props.sort(key=lambda e: e[i])
    return tuple(bonds_props)


def find_mol_sym_atoms(mol : Mol) -> dict:
    """
    Find symmetric atoms in molecule

    Parameters
    ----------
    mol : Mol
        Molecule to find symmetric atoms

    Returns
    -------
    sym_atoms : list of sets of integers
        List of sets of symmetric atoms
    """
    sym_atoms = []

    # Step 1: Initialize list of sets of symmetric atoms using atoms properties only
    atoms_props = {}
    for atom in mol.GetAtoms():
        atom_key = get_atom_props(atom, ring_size=mol.GetNumAtoms())
        if atom_key in atoms_props:
            atoms_props[atom_key].add(atom)
        else:
            atoms_props[atom_key] = set({atom})
    sym_atoms = list(atoms_props.values())
    atoms_to_sym_cls = {
        atom.GetIdx() : i for i, cls in enumerate(sym_atoms) for atom in cls
    }
    
    # Step 2: Cut sets of symmetric atoms until atoms have different bonds properties
    while True:
        is_break = True
        for cls_idx, cls in enumerate(sym_atoms):
            if len(cls) > 1:
                cls_atoms_props = {}
                for atom in cls:
                    atom_key = get_atom_bonds_props(atom, atoms_to_sym_cls)
                    if atom_key in cls_atoms_props:
                        cls_atoms_props[atom_key].add(atom)
                    else:
                        cls_atoms_props[atom_key] = set({atom})
                if len(cls_atoms_props) > 1:
                    cls_sym_atoms = list(cls_atoms_props.values())
                    for i in range(1,len(cls_sym_atoms)):
                        subcls = cls_sym_atoms[i]
                        subcls_idx = len(sym_atoms)
                        cls = cls - subcls
                        sym_atoms.append(subcls)
                        for atom in subcls:
                            atoms_to_sym_cls[atom.GetIdx()] = subcls_idx
                    sym_atoms[cls_idx] = cls
                    is_break = False
        if is_break:
            break
    
    return { a.GetIdx() : set({ b.GetIdx() for b in cls }) 
             for cls in sym_atoms for a in cls }


def submolecule(
    mol: Mol,
    atoms: FrozenSet[int],
    atom_maps: List[int] | None = None,
    is_connect: bool = False,
) -> Mol:
    """
    Build a submolecule induced by a set of atom indices.

    Behavior
    --------
    - Considers only bonds with both endpoints inside `atoms`.
      Isolated atoms are intentionally omitted.
    - If the induced subgraph splits into multiple connected components:
      - when `is_connect` is True, returns an empty Mol (drop case);
      - when True, returns a single representative component (the first one).

    Parameters
    ----------
    mol : Mol
        Input RDKit molecule.
    atoms : FrozenSet[int]
        Original atom indices to induce the submolecule on.
    atom_maps : List[int] | None
        Optional out mapping: result atom index -> original atom index.
        Populated only when a non-empty submolecule is returned.
    is_connect : bool
        Reject disconnected induced submolecules. If False, the first
        connected fragment is returned instead of dropping the result.

    Returns
    -------
    Mol
        Non-empty connected submolecule or empty Mol if dropped.
    """
    # Collect bond indices fully contained within the atom subset via
    # local neighborhoods (linear in the subset degree sum).
    bond_ids = set()
    for i in atoms:
        ai = mol.GetAtomWithIdx(i)
        for b in ai.GetBonds():
            j = b.GetOtherAtomIdx(i)
            if j in atoms:
                bond_ids.add(b.GetIdx())

    # No internal bonds -> induced subgraph has no edges; return empty Mol
    if not bond_ids:
        return Mol()

    # Build the induced substructure (may be disconnected). Also capture a
    # mapping from `sub` atom indices to original `mol` atom indices.
    atom_map_sub = {}
    sub = PathToSubmol(mol, list(bond_ids), atomMap=atom_map_sub)
    atom_map_sub = {v: k for k, v in atom_map_sub.items()}

    # Obtain connected components as atom index tuples (fast, no sanitize)
    frag_atom_maps = []
    frags = GetMolFrags(sub, asMols=True, sanitizeFrags=False,
                        fragsMolAtomMapping=frag_atom_maps)

    # Drop disconnected results when requested
    if is_connect and len(frags) > 1:
        return Mol()

    # Select the first connected fragment
    res = frags[0]

    # Fill `atom_maps`: result atom index -> original atom index in `mol`
    if atom_maps is not None and frag_atom_maps:
        frag_to_sub = frag_atom_maps[0]
        atom_maps[:] = [atom_map_sub[sub_idx] for sub_idx in frag_to_sub]

    # Heuristic: demote non-ring aromatic atoms/bonds to reduce sanitize errors
    for a in res.GetAtoms():
        if (not a.IsInRing()) and a.GetIsAromatic():
            a.SetIsAromatic(False)
    for b in res.GetBonds():
        if (not b.IsInRing()) and b.GetIsAromatic():
            b.SetIsAromatic(False)
            # After removing off-ring aromaticity, use a single bond by default
            b.SetBondType(BondType.SINGLE)

    # Sanitize for chemistry consistency; on failure, retry without kekulization
    try:
        SanitizeMol(res)
    except Exception:
        try:
            sanitize_ops = (
                SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
            )
            SanitizeMol(res, sanitizeOps=sanitize_ops)
        except Exception:
            # As a last resort, return the unsanitized fragment
            pass

    return res


def to_graph(mol : Mol) -> nx.Graph:
    """
    Converts an RDKit molecule (Mol) to a NetworkX graph.

    Parameters
    ----------
    mol : Mol
        The RDKit molecule to convert.

    Returns
    -------
    res : nx.Graph
        The NetworkX graph representation of the molecule.
    """
    graph = nx.Graph()

    # Add atoms as nodes
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx())

    # Add bonds as edges
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph
