# External imports
import torch
from rdkit.Chem.rdchem import Mol
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple
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

        atom_ids = tuple(range(atom_num))
        root_coalition = frozenset(atom_ids)

        # Maintain BFS traversal state without materializing the full DAG.
        node_molecules: List[Optional[Mol]] = [mol]
        node_orig_atoms: List[Tuple[int, ...]] = [atom_ids]
        node_predictions: List[Optional[torch.Tensor]] = []
        queue = deque([0])
        coalition_to_id: Dict[frozenset[int], int] = {root_coalition: 0}

        root_pred = self._predict_batch([mol])[0]
        node_predictions.append(root_pred)

        atom_scores = torch.zeros(atom_num, dtype=root_pred.dtype, device=self.device)
        atom_factor = torch.zeros(atom_num, dtype=root_pred.dtype, device=self.device)

        while queue:
            node_id = queue.popleft()
            parent_pred = node_predictions[node_id]
            cur_mol = node_molecules[node_id]
            orig_atoms = node_orig_atoms[node_id]

            if parent_pred is None or cur_mol is None:
                continue

            edges, new_child_ids = self._coalesce_children(
                node_id,
                node_molecules,
                node_orig_atoms,
                node_predictions,
                coalition_to_id,
                queue,
            )

            if not edges:
                node_molecules[node_id] = None
                continue

            if new_child_ids:
                new_preds = self._predict_batch(
                    [node_molecules[child_id] for child_id in new_child_ids]
                )
                for child_id, pred in zip(new_child_ids, new_preds):
                    node_predictions[child_id] = pred

            for child_id, removed_atoms in edges:
                child_pred = node_predictions[child_id]
                if child_pred is None or not removed_atoms:
                    continue

                if self.shapley_mode == 'absolute':
                    score = parent_pred - child_pred

                elif self.shapley_mode == 'relative':
                    score = 1 - child_pred / parent_pred

                else:
                    score = parent_pred.new_zeros(())

                atoms_idx = list(removed_atoms)
                atom_scores[atoms_idx] += score / len(removed_atoms)
                atom_factor[atoms_idx] += 1

            # Release the RDKit molecule to reduce memory pressure
            node_molecules[node_id] = None

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

        node_molecules: List[Optional[Mol]] = [mol]
        node_orig_atoms: List[Tuple[int, ...]] = [atom_ids]
        node_predictions: List[Optional[torch.Tensor]] = [None]
        stack = deque([0])
        coalition_to_id = {root: 0}

        parent_ids: List[int] = []
        child_ids: List[int] = []
        edge_atom_indices: List[int] = []
        edge_atom_counts: List[int] = []

        while stack:
            node_id = stack.popleft()
            edges, _ = self._coalesce_children(
                node_id,
                node_molecules,
                node_orig_atoms,
                node_predictions,
                coalition_to_id,
                stack,
            )

            for child_id, removed_atoms in edges:
                parent_ids.append(node_id)
                child_ids.append(child_id)
                edge_atom_indices.extend(removed_atoms)
                edge_atom_counts.append(len(removed_atoms))

        return _CoalitionGraph(
            molecules=tuple(node_molecules),
            parent_ids=tuple(parent_ids),
            child_ids=tuple(child_ids),
            edge_atom_indices=tuple(edge_atom_indices),
            edge_atom_counts=tuple(edge_atom_counts),
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

    def _coalesce_children(
        self,
        node_id: int,
        node_molecules: List[Optional[Mol]],
        node_orig_atoms: List[Tuple[int, ...]],
        node_predictions: List[Optional[torch.Tensor]],
        coalition_to_id: Dict[frozenset[int], int],
        queue: deque[int],
    ) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[int]]:
        """Expand a coalition node and return its children and new node ids."""
        cur_mol = node_molecules[node_id]
        if cur_mol is None:
            return [], []

        orig_atoms = node_orig_atoms[node_id]
        atom_ids = range(cur_mol.GetNumAtoms())

        sym_classes = symmetry_classes(cur_mol, self.symmetry)
        aps = articulation_classes(cur_mol, sym_classes)
        all_nodes_id = frozenset(atom_ids)

        edges: List[Tuple[int, Tuple[int, ...]]] = []
        new_child_ids: List[int] = []

        for cls_idx, cls_atoms in enumerate(sym_classes):
            if cls_idx in aps:
                continue

            if cur_mol.GetNumAtoms() < len(cls_atoms) + self.min_atoms:
                continue

            child_nodes_id = all_nodes_id.difference(cls_atoms)
            child_coalition = frozenset(orig_atoms[i] for i in child_nodes_id)
            removed_atoms = tuple(orig_atoms[i] for i in cls_atoms)

            if child_coalition in coalition_to_id:
                child_id = coalition_to_id[child_coalition]
                edges.append((child_id, removed_atoms))
                continue

            atom_maps: List[int] = []
            child_mol = submolecule(cur_mol, child_nodes_id, atom_maps)
            if child_mol.GetNumAtoms() < self.min_atoms:
                continue

            child_id = len(node_molecules)
            coalition_to_id[child_coalition] = child_id
            node_molecules.append(child_mol)
            node_orig_atoms.append(tuple(orig_atoms[i] for i in atom_maps))
            node_predictions.append(None)
            queue.append(child_id)
            edges.append((child_id, removed_atoms))
            new_child_ids.append(child_id)

        return edges, new_child_ids
