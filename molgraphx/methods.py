# External imports
import torch
import networkx as nx
from rdkit.Chem.rdchem import Mol
from typing import Callable, List, Iterable
from collections import deque

# Internal imports
from molgraphx.molecule import symmetry_classes, articulation_classes, submolecule


SHAPLEY_MODES = set({'absolute', 'relative'})

    
class AtomsExplainer(object):
    """Atom-level explanations for molecular graphs."""

    def __init__(
        self,
        predictor: Callable[[Iterable[Mol]], torch.Tensor],
        min_atoms: int = 5,
        symmetry: bool = True,
        balanced: bool = True,
        shapley_mode : str = 'absolute',
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the atoms explainer.

        Parameters
        ----------
        predictor : Callable[[Iterable[Mol]], torch.Tensor]
            Callable that maps a batch (iterable) of RDKit molecules to target values.
        min_atoms : int
            Minimum number of atoms allowed in any submolecule (stopping criterion).
        symmetry : bool
            Whether to account for molecular symmetry (group atoms into classes).
        balanced : bool
            If True, average per-atom contributions by the number of times
            an atom appears across edges.
        shapley_mode : str
            Atom contribution mode:
            - 'absolute': parent_pred - child_pred
            - 'relative': 1 - child_pred / parent_pred
        """

        self.predictor = predictor
        assert min_atoms > 0, \
               f"Bad value '{min_atoms}' for parameter min_atoms."
        self.min_atoms = min_atoms if min_atoms > 0 else 1
        self.symmetry = symmetry
        self.balanced = balanced
        assert shapley_mode in SHAPLEY_MODES, \
               f"Bad value '{shapley_mode}' for parameter shapley_mode."
        self.shapley_mode = shapley_mode
        self.device = device

    def __call__(self, mol: Mol) -> torch.Tensor:
        """
        Compute atom prediction scores for a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule whose atoms need to be explained.

        Returns
        -------
        List[float]
            Scores for atoms of the input molecule (same indexing as `mol`).
        """

        atom_num = mol.GetNumAtoms()

        # Skip small molecules
        if atom_num <= self.min_atoms:
            return (self.predictor([mol]) / atom_num).repeat(atom_num)
        
        # Build the coalition DAG that organizes connected submolecules
        graph = self._build_coalition_graph(mol)

        # Batched predictions for all submolecules in the coalition graph
        coalitions, mols = zip(*graph.nodes(data="molecule"))
        predictions = { coalitions[i] : p for i, p in enumerate(self.predictor(mols)) }

        atom_scores = torch.zeros(atom_num, dtype=torch.float, device=self.device)
        atom_factor = torch.zeros(atom_num, dtype=torch.int, device=self.device)

        # Accumulate per-atom contributions over coalition graph edges
        for u, v, atoms in graph.edges(data='atoms'):
            
            if self.shapley_mode == 'absolute':
                score = predictions[u] - predictions[v]

            elif self.shapley_mode == 'relative':
                score = 1 - predictions[v] / predictions[u]

            else:
                score = 0

            atom_scores[atoms] += score / len(atoms)
            atom_factor[atoms] += 1

        # Normalize by the number of times each atom was updated
        if self.balanced:
            atom_scores /= atom_factor

        return atom_scores

    # --- Internal helpers -------------------------------------------------

    def _build_coalition_graph(self, mol: Mol) -> nx.DiGraph:
        """
        Build the coalition DAG for the given molecule.

        - Nodes: coalitions (frozenset of original atom indices). The node
          stores a connected representative submolecule in the "molecule" attribute.
        - Edges: parent -> child if the child’s molecule is obtained from the
          parent’s by removing exactly one symmetry class that does not break
          connectivity of the factor graph.

        Node attributes
        ---------------
        - "molecule": RDKit Mol of the connected submolecule.
        - "orig_atoms": list[int] mapping submolecule atom index -> original
          atom index in the input molecule.

        The root coalition contains all atom indices of the input molecule.
        """
        atom_num = mol.GetNumAtoms()
        atom_ids = range(atom_num)
        root = frozenset(atom_ids)
        stack = deque([root])

        graph = nx.DiGraph()
        graph.add_node(root, molecule=mol, orig_atoms=tuple(atom_ids))

        while len(stack) > 0:
            coalition = stack.popleft()
            node_attrs = graph.nodes[coalition]
            cur_mol = node_attrs["molecule"]
            orig_atoms = node_attrs["orig_atoms"]
            atom_ids = range(cur_mol.GetNumAtoms())

            # Find symmetry classes of atoms in cur_mol
            sym_classes = symmetry_classes(cur_mol, self.symmetry)

            # Classes whose removal disconnects the molecule cur_mol
            aps = articulation_classes(cur_mol, sym_classes)

            # Full set of atom indices for cur_mol (for difference operations)
            all_nodes_id = frozenset(atom_ids)

            # Iterate classes (potential removals)
            for cls_idx, cls_atoms in enumerate(sym_classes):
                # Keep the molecule connected after removal
                if cls_idx in aps:
                    continue

                # Enforce size threshold on the child submolecule
                if cur_mol.GetNumAtoms() < len(cls_atoms) + self.min_atoms:
                    continue

                # Child coalition (original indexing)
                node_atoms = tuple(cls_atoms)
                child_nodes_id = all_nodes_id.difference(node_atoms)
                child_coalition = frozenset(orig_atoms[i] for i in child_nodes_id)

                # Already seen this coalition — add edge and skip
                if graph.has_node(child_coalition):
                    graph.add_edge(coalition, child_coalition, 
                                   atoms=tuple(orig_atoms[i] for i in node_atoms))
                    continue

                # Build child submolecule and map back to original atoms
                atom_maps: List[int] = []
                child_mol = submolecule(cur_mol, child_nodes_id, atom_maps)
                if child_mol.GetNumAtoms() < self.min_atoms:
                    continue

                # Register node and edge; enqueue child
                graph.add_node(child_coalition, molecule=child_mol, 
                               orig_atoms=tuple(orig_atoms[i] for i in atom_maps))
                graph.add_edge(coalition, child_coalition, 
                               atoms=tuple(orig_atoms[i] for i in node_atoms))
                stack.append(child_coalition)

        return graph
