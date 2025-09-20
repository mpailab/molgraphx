# External imports
import rdkit
import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from rdkit.Chem.rdchem import Mol
from typing import Callable, List, Tuple, FrozenSet, Set, Dict, Generator
from collections import deque

# Internal imports
from molgraphx.molecule import find_mol_sym_atoms, submolecule, to_graph

# Type aliases
PredictorFn = Callable[[Generator[Mol, None, None] | List[Mol] | Tuple[Mol]], torch.Tensor]

SHAPLEY_MODES = set({'absolute', 'relative'})
    
class AtomsExplainer(object):
    """Explain atoms of molecular graphs"""

    def __init__(
        self,
        predictor: PredictorFn,
        min_atoms: int = 5,
        symmetry: bool = True,
        balanced: bool = True,
        shapley_mode : str = 'absolute',
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize atoms explainer.

        Parameters
        ----------
        predictor : PredictorFn
            Function mapping a batch of molecules to target values.
        min_atoms : int
            Minimal number of atoms to continue splitting submolecules.
        symmetry : bool
            Whether to taken into account molecular symmetry.
        shapley_mode : str
            Mode for calculating Shapley values of atoms. 
            Possible values:
            - 'absolute': ...
            - 'relative': ...
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
            return (self.predictor([mol]) / atom_num).repeat(atom_num)
        
        # Build the coalition graph (DAG) that organizes submolecules
        graph = self._build_coalition_graph(mol)

        # Batched predictions for all submolecules in the coalition graph
        
        coalitions, mols = zip(*graph.nodes(data="molecule"))
        predictions = { coalitions[i] : p for i, p in enumerate(self.predictor(mols)) }

        atom_scores = torch.zeros(atom_num, dtype=torch.float, device=self.device)
        atom_factor = torch.zeros(atom_num, dtype=torch.int, device=self.device)

        for u, v, atoms in graph.edges(data='atoms'):
            
            if self.shapley_mode == 'absolute':
                score = predictions[u] - predictions[v]

            elif self.shapley_mode == 'relative':
                score = 1 - predictions[v] / predictions[u]

            else:
                score = 0

            atom_scores[atoms] += score / len(atoms)
            atom_factor[atoms] += 1

        if self.balanced:
            atom_scores /= atom_factor

        return atom_scores

        # atom_num = mol.GetNumAtoms()

        # # Skip small molecules
        # if atom_num <= self.min_atoms:
        #     return (self.predictor([mol]) / atom_num).repeat(atom_num)
        
        # # Build the coalition graph (DAG) that organizes submolecules
        # graph = self._build_coalition_graph(mol)
        # data = from_networkx(graph, group_edge_attrs=['atoms'])

        # # Batched predictions for all submolecules in the coalition graph
        # predictions = self.predictor(mol for _, mol in graph.nodes(data="molecule"))

        # if self.shapley_mode == 'absolute':
        #     scores = predictions[data.edge_index[0]] - predictions[data.edge_index[1]]

        # elif self.shapley_mode == 'relative':
        #     scores = 1 - predictions[data.edge_index[1]] / predictions[data.edge_index[0]]

        # else:
        #     scores = None

        # scores /= data.edge_attr.sum(dim=1)
        # print(scores.size(), scores)
        # print(data.edge_attr.size(), data.edge_attr)
        # atom_scores = (scores * data.edge_attr).sum(dim=1)

        # if self.balanced:
        #     atom_scores /= data.edge_attr.sum(dim=1)

        # return atom_scores

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
        - "orig_atoms": list[list[int]] mapping submolecule atom index -> list of
          original atom indices in the input molecule.
        - "prediction": model output for this submolecule.
        - "synergy": contribution of this coalition to the model output 
          exclusive of all successors.
        - "successors": set of all nodes reachable from a given node.

        The root coalition contains all atom indices of the input molecule.
        """
        atom_num = mol.GetNumAtoms()
        atom_ids = range(atom_num)
        root = frozenset(atom_ids)
        stack = deque([root])

        graph = nx.DiGraph()
        graph.add_node(root, molecule=mol, orig_atoms=list(atom_ids))

        while len(stack) > 0:
            coalition = stack.popleft()
            node_attrs = graph.nodes[coalition]
            cur_mol = node_attrs["molecule"]
            orig_atoms = node_attrs["orig_atoms"]
            atom_ids = range(cur_mol.GetNumAtoms())

            # print("\n---------------------------------------------------")
            # print(coalition, cur_mol.GetNumAtoms())

            # Atom equivalence relation (symmetry-aware if requested)
            if self.symmetry:
                sym_atoms = find_mol_sym_atoms(cur_mol)
            else:
                sym_atoms = [ set({i}) for i in atom_ids ]

            # Factor (quotient) graph for the current submolecule
            cur_graph = to_graph(cur_mol)
            # print(cur_graph.nodes)
            fac_graph = nx.quotient_graph(cur_graph, sym_atoms)

            # Articulation points: removing them disconnects the factor graph
            aps = set(nx.articulation_points(fac_graph))

            # Full set of atom indices for cur_mol (for quick difference)
            all_nodes_id = frozenset(atom_ids)

            # Iterate all nodes in the current factor graph
            for node in fac_graph.nodes:

                # print(node, node in aps, cur_mol.GetNumAtoms() < len(node) + self.min_atoms)

                # Keep the factor graph connected in the child
                if node in aps:
                    continue

                # Size threshold on child submolecule
                if cur_mol.GetNumAtoms() < len(node) + self.min_atoms:
                    continue

                # Child coalition (original indexing)
                child_nodes_id = all_nodes_id - node
                child_coalition = frozenset({orig_atoms[i] for i in child_nodes_id})

                # Already seen this coalition — skip duplicate  
                if child_coalition in graph.nodes:
                    graph.add_edge(coalition, child_coalition,
                                   atoms=[orig_atoms[i] for i in node])
                    # print(coalition, " -> ", child_coalition)
                    continue

                # Build child submolecule and map back to original atoms
                atom_maps = []
                child_mol = submolecule(cur_mol, child_nodes_id, atom_maps)
                if child_mol.GetNumAtoms() < self.min_atoms:
                    continue

                # Map child atoms back to original atom sets.
                child_orig_atoms = [ orig_atoms[i] for i in atom_maps ]

                # Register the submolecule node and enqueue it for further expansion
                graph.add_node(child_coalition, 
                               molecule=child_mol, 
                               orig_atoms=child_orig_atoms)
                graph.add_edge(coalition, child_coalition,
                               atoms=[orig_atoms[i] for i in node])
                stack.append(child_coalition)
                # print(coalition, " -> ", child_coalition)

        return graph
