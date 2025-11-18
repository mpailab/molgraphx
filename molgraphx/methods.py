# External imports
import torch
from rdkit.Chem.rdchem import Mol
from typing import Callable, Dict, Iterable, List, NamedTuple, Tuple
from collections import deque

# Internal imports
from molgraphx.molecule import symmetry_classes, articulation_classes, submolecule


SHAPLEY_MODES = set({'absolute', 'relative'})

class _CoalitionGraph(NamedTuple):
    """Lightweight storage for coalition DAG nodes and edges."""
    molecules: Tuple[Mol, ...]
    parent_ids: Tuple[int, ...]
    child_ids: Tuple[int, ...]
    edge_atom_indices: Tuple[int, ...]
    edge_atom_counts: Tuple[int, ...]
    edge_atom_edge_ids: Tuple[int, ...]


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
        predictions = self._predict_batch(graph.molecules)

        atom_scores = torch.zeros(atom_num, dtype=predictions.dtype, device=self.device)
        atom_factor = torch.zeros(atom_num, dtype=predictions.dtype, device=self.device)

        if graph.parent_ids:
            parent_ids = torch.tensor(graph.parent_ids, dtype=torch.long, device=self.device)
            child_ids = torch.tensor(graph.child_ids, dtype=torch.long, device=self.device)

            parent_pred = predictions[parent_ids]
            child_pred = predictions[child_ids]

            if self.shapley_mode == 'absolute':
                edge_scores = parent_pred - child_pred

            elif self.shapley_mode == 'relative':
                edge_scores = 1 - child_pred / parent_pred

            else:
                edge_scores = torch.zeros_like(parent_pred)

            edge_sizes = torch.tensor(
                graph.edge_atom_counts,
                dtype=edge_scores.dtype,
                device=self.device,
            )
            edge_contrib = edge_scores / edge_sizes

            if graph.edge_atom_indices:
                atom_indices_tensor = torch.tensor(
                    graph.edge_atom_indices, dtype=torch.long, device=self.device
                )
                edge_index_tensor = torch.tensor(
                    graph.edge_atom_edge_ids, dtype=torch.long, device=self.device
                )
                per_atom_scores = edge_contrib[edge_index_tensor]
                atom_scores.index_add_(0, atom_indices_tensor, per_atom_scores)
                atom_factor.index_add_(
                    0, atom_indices_tensor, torch.ones_like(per_atom_scores)
                )

        # Normalize by the number of times each atom was updated
        if self.balanced:
            mask = atom_factor > 0
            atom_scores[mask] /= atom_factor[mask]

        return atom_scores

    # --- Internal helpers -------------------------------------------------

    def _build_coalition_graph(self, mol: Mol) -> _CoalitionGraph:
        """
        Build the coalition DAG for the given molecule.

        - Nodes: coalitions (frozenset of original atom indices). Each node
          stores a connected representative submolecule.
        - Edges: parent -> child if the child’s molecule is obtained from the
          parent’s by removing exactly one symmetry class that does not break
          connectivity of the factor graph.

        Returns
        -------
        _CoalitionGraph
            Compact storage of node molecules and the edges between them.

        The root coalition contains all atom indices of the input molecule.
        """
        atom_num = mol.GetNumAtoms()
        atom_ids = tuple(range(atom_num))
        root = frozenset(atom_ids)

        node_molecules: List[Mol] = [mol]
        node_orig_atoms: List[Tuple[int, ...]] = [atom_ids]
        stack = deque([0])
        coalition_to_id: Dict[frozenset[int], int] = {root: 0}

        parent_ids: List[int] = []
        child_ids: List[int] = []
        edge_atom_indices: List[int] = []
        edge_atom_counts: List[int] = []
        edge_atom_edge_ids: List[int] = []

        while stack:
            node_id = stack.popleft()
            cur_mol = node_molecules[node_id]
            orig_atoms = node_orig_atoms[node_id]
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

                child_nodes_id = all_nodes_id.difference(cls_atoms)
                child_coalition = frozenset(orig_atoms[i] for i in child_nodes_id)
                removed_atoms = tuple(orig_atoms[i] for i in cls_atoms)

                if child_coalition in coalition_to_id:
                    child_id = coalition_to_id[child_coalition]
                else:
                    atom_maps: List[int] = []
                    child_mol = submolecule(cur_mol, child_nodes_id, atom_maps)
                    if child_mol.GetNumAtoms() < self.min_atoms:
                        continue

                    child_id = len(node_molecules)
                    coalition_to_id[child_coalition] = child_id
                    node_molecules.append(child_mol)
                    node_orig_atoms.append(tuple(orig_atoms[i] for i in atom_maps))
                    stack.append(child_id)

                edge_idx = len(parent_ids)
                parent_ids.append(node_id)
                child_ids.append(child_id)
                edge_atom_indices.extend(removed_atoms)
                edge_atom_counts.append(len(removed_atoms))
                edge_atom_edge_ids.extend([edge_idx] * len(removed_atoms))

        return _CoalitionGraph(
            molecules=tuple(node_molecules),
            parent_ids=tuple(parent_ids),
            child_ids=tuple(child_ids),
            edge_atom_indices=tuple(edge_atom_indices),
            edge_atom_counts=tuple(edge_atom_counts),
            edge_atom_edge_ids=tuple(edge_atom_edge_ids),
        )

    def _predict_batch(self, mols: Iterable[Mol]) -> torch.Tensor:
        """Run the predictor and coerce the result to a 1-D tensor on device."""
        mols = list(mols)
        if not mols:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        outputs = self.predictor(mols)
        if isinstance(outputs, torch.Tensor):
            tensor = outputs
        else:
            tensor = torch.as_tensor(outputs)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 1:
            if any(dim != 1 for dim in tensor.shape[1:]):
                raise ValueError("predictor must return a scalar per molecule.")
            tensor = tensor.reshape(tensor.shape[0])

        return tensor.to(self.device)
