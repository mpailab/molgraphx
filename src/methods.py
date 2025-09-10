# External imports
import rdkit
import torch
import networkx as nx
from rdkit.Chem.rdchem import Mol
from typing import Callable, List, Tuple, FrozenSet, Set, Dict
from collections import deque

# Internal imports
from molecule import find_mol_sym_atoms, submolecule, to_graph

# Type aliases
PredictorFn = Callable[[List[Mol] | Tuple[Mol]], List[float]]
    
class AtomsExplainer(object):
    """Explain atoms of molecular graphs"""

    def __init__(
        self,
        predictor: PredictorFn,
        min_atoms: int = 5,
        is_sym: bool = True,
        allow_disconnected: bool = False,
    ):
        """
        Initialize atoms explainer.

        Parameters
        ----------
        predictor : PredictorFn
            Function mapping a batch of molecules to target values.
        min_atoms : int
            Minimal number of atoms to continue splitting submolecules.
        is_sym : bool
            Whether to taken into account molecular symmetry.
        allow_disconnected : bool
            If True, allow coalitions of original atoms that induce a
            disconnected molecule (number of fragments > 1).
        """

        self.predictor = predictor
        self.min_atoms = min_atoms
        self.is_sym = is_sym
        self.allow_disconnected = allow_disconnected

    def __call__(self, mol: Mol) -> List[float]:
        """
        Explain atoms for a molecule.

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
            atom_scores = [ self.predictor([mol])[0] / atom_num ] * atom_num
            return atom_scores
        
        # Build the coalition graph (DAG) that organizes submolecules
        graph = self._build_coalition_graph(mol)

        # Batched predictions for all submolecules in the coalition graph
        gen = ((c, n["molecule"]) for c, n in graph.nodes.items())
        coalitions, mols = zip(*gen)
        for i, p in enumerate(self.predictor(mols)):
            node = graph.nodes[coalitions[i]]
            node["prediction"] = p

        atom_scores = [0.0] * atom_num

        # Reverse topological order (bottom-up)
        for coalition in reversed(list(nx.topological_sort(graph))):
            node = graph.nodes[coalition]

            # Transitive closure of successors:
            # union of direct children and their successors
            node["successors"] = set()
            for succ_coalition in graph.successors(coalition):
                succ = graph.nodes[succ_coalition]
                node["successors"] |= set({succ_coalition}) | succ["successors"]

            # Synergy = node's prediction minus the synergies of all its successors
            node["synergy"] = node["prediction"]
            for succ_coalition in node["successors"]:
                succ = graph.nodes[succ_coalition]
                node["synergy"] -= succ["synergy"]

            # Distribute synergy uniformly over atoms of the current coalition
            # (normalize by the number of atoms in the current submolecule)
            share = node["synergy"] / node["molecule"].GetNumAtoms()
            for i in coalition:
                atom_scores[i] += share

        return atom_scores

    def _build_coalition_graph(self, mol: Mol) -> nx.DiGraph:
        """
        Build the coalition graph (DAG) for the given molecule.

        - Nodes: coalitions (frozenset of original atom indices). The induced
          subgraph can split into disconnected but equivalent components; we
          keep one representative connected component in the "molecule"
          attribute.
        - Edges: parent -> child if the child’s molecule is a submolecule of
          the parent’s (obtained by removing exactly one factor-graph node).

        Node attributes
        ---------------
        - "molecule": RDKit Mol of the representative connected submolecule.
        - "orig_atoms": list[set[int]] mapping submolecule atom index -> set of
          original atom indices in the input molecule.
        - "prediction": model output for this submolecule.
        - "synergy": contribution of this coalition to the model output 
          exclusive of all successors.
        - "successors": set of all nodes reachable from a given node.

        The root coalition contains all atom indices of the input molecule.
        """
        atom_ids = range(mol.GetNumAtoms())
        root = frozenset(atom_ids)
        stack = deque([root])
        graph = nx.DiGraph()
        graph.add_node(root, molecule=mol, orig_atoms=[set({i}) for i in atom_ids])

        while len(stack) > 0:
            coalition = stack.popleft()
            node_attrs = graph.nodes[coalition]
            cur_mol = node_attrs["molecule"]
            orig_atoms = node_attrs["orig_atoms"]

            # Atom equivalence relation (symmetry-aware if requested)
            if self.is_sym:
                sym_atoms = find_mol_sym_atoms(cur_mol)
                atom_relation = lambda x, y: x in sym_atoms[y]
            else:
                atom_relation = lambda x, y: x == y

            # Factor (quotient) graph for the current submolecule
            cur_graph = to_graph(cur_mol)
            fac_graph = nx.quotient_graph(cur_graph, atom_relation)

            # Articulation points: removing them disconnects the factor graph
            aps = set(nx.articulation_points(fac_graph))

            # Full set of atom indices for cur_mol (for quick difference)
            all_nodes_id = frozenset(range(cur_mol.GetNumAtoms()))

            # Iterate all nodes in the current factor graph
            for node in fac_graph.nodes:
                # Keep the factor graph connected in the child
                if node in aps:
                    continue
                # Size threshold on child submolecule
                if cur_mol.GetNumAtoms() < len(node) + self.min_atoms:
                    continue

                # Child coalition (original indexing)
                child_nodes_id = all_nodes_id - node
                child_coalition = frozenset({j for i in child_nodes_id 
                                               for j in orig_atoms[i]})

                # Already seen this coalition — skip duplicate
                if child_coalition in graph.nodes:
                    continue

                # Build child submolecule and map back to original atoms
                atom_maps = []
                child_mol = submolecule(cur_mol, child_nodes_id, atom_maps,
                                        allow_disconnected=self.allow_disconnected)
                if child_mol.GetNumAtoms() < self.min_atoms:
                    continue

                # Map child atoms back to original atom sets.
                if self.is_sym:
                    sym_atoms = find_mol_sym_atoms(cur_mol)
                    child_orig_atoms = [
                        set({a for j in sym_atoms[atom_maps[i]] for a in orig_atoms[j]})
                        for i in range(child_mol.GetNumAtoms())
                    ]
                else:
                    child_orig_atoms = [
                        set(orig_atoms[atom_maps[i]])
                        for i in range(child_mol.GetNumAtoms())
                    ]

                # Register the submolecule node and enqueue it for further expansion
                graph.add_node(child_coalition, 
                               molecule=child_mol, 
                               orig_atoms=child_orig_atoms)
                graph.add_edge(coalition, child_coalition)
                stack.append(child_coalition)

        return graph
