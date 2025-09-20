# External imports
from rdkit.Chem import SanitizeMol, GetMolFrags, PathToSubmol
from rdkit.Chem.rdmolops import SanitizeFlags
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
from rdkit.Chem.rdchem import BondType, Mol
from typing import FrozenSet, Dict, List, Set


def symmetry_classes(mol: Mol, symmetry: bool = True) -> List[Set[int]]:
    """Return symmetry classes (sets of atom indices) for ``mol``.

    Uses RDKit's canonical ranking to group atoms by rank (fast, C++).
    If ``symmetry`` is False or ranking fails, returns singleton classes.
    """
    if not symmetry:
        return [set({i}) for i in range(mol.GetNumAtoms())]

    try:
        groups = CanonicalRankAtoms(mol, breakTies=True)
        cls: Dict[int, Set[int]] = {}
        for i, g in enumerate(groups):
            cls.setdefault(int(g), set()).add(i)
        return [cls[k] for k in sorted(cls.keys())]

    except Exception:
        # Fallback to trivial partition if RDKit ranking fails
        return [set({i}) for i in range(mol.GetNumAtoms())]


def articulation_classes(mol: Mol, classes: List[Set[int]]) -> Set[int]:
    """Return indices of symmetry classes that are articulation points.

    A class is an articulation point if removing all atoms of that class
    disconnects the factor graph of classes.
    """

    # Build undirected adjacency of symmetry classes
    cls_of = {}
    for ci, cls in enumerate(classes):
        for a in cls:
            cls_of[a] = ci
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(classes))}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ci, cj = cls_of[i], cls_of[j]
        if ci != cj:
            adj[ci].add(cj)
            adj[cj].add(ci)

    # Find articulation points in the undirected class graph
    time = 0                      # Global DFS timestamp (order of vertex discovery)
    disc: Dict[int, int] = {}     # Discovery time for each vertex
    low: Dict[int, int] = {}      # Lowest discovery time reachable from the subtree (low-link)
    parent: Dict[int, int | None] = {}  # Parent in the DFS tree
    visited: Set[int] = set()     # Set of visited vertices
    aps: Set[int] = set()         # Articulation points

    def dfs(u: int, root: int) -> int:
        nonlocal time
        visited.add(u)
        time += 1
        disc[u] = low[u] = time # Initialize discovery time and low value
        parent.setdefault(u, None)
        children = 0
        for v in adj.get(u, ()):  # neighbors
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v, root)

                # Propagate low-link from the subtree
                low[u] = min(low[u], low[v])

                # Non-root: child subtree cannot reach an ancestor of u
                if parent[u] is not None and low[v] >= disc[u]:
                    aps.add(u)

            elif v != parent[u]:
                # Back-edge: update low-link of u from ancestor v
                low[u] = min(low[u], disc[v])

        # Root rule: root with more than one DFS child is an articulation point
        if parent[u] is None and children > 1:
            aps.add(u)

        return children

    for u in adj.keys():
        if u not in visited:
            dfs(u, u)

    return aps


def submolecule(
    mol: Mol,
    atoms: FrozenSet[int],
    atom_maps: List[int] | None = None,
) -> Mol:
    """Build a connected submolecule induced by a set of atom indices.

    Behavior
    --------
    - Only bonds with both endpoints inside ``atoms`` are considered.
      Isolated atoms are omitted on purpose.
    - If the induced subgraph splits into multiple connected components,
      the result is dropped (returns an empty Mol).

    Parameters
    ----------
    mol : Mol
        Input RDKit molecule.
    atoms : FrozenSet[int]
        Original atom indices to induce the submolecule on.
    atom_maps : List[int] | None
        Optional out mapping: result atom index -> original atom index.
        Populated only when a non-empty submolecule is returned.

    Returns
    -------
    Mol
        Non-empty connected submolecule, empty Mol if dropped due to
        disconnection or sanitization failure.
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

    # Drop disconnected results (always enforce connected output)
    if len(frags) > 1:
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
            # As a last resort, return an empty Mol on failure
            return Mol()

    return res
