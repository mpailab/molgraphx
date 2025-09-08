# External imports
import rdkit
import torch
import networkx as nx
from rdkit.Chem.rdchem import Mol
from typing import Callable, List, FrozenSet

# Internal imports
from .molecule import find_mol_sym_atoms, submolecule, to_graph
from .utils import coalition_id, weak_shapley_score

    
class AtomsExplainer(object):
    """Explain atoms of molecular graphs"""

    def __init__(self, 
                 predictor: Callable[[List[Mol]], List[float]],
                 shapley_score: Callable[[FrozenSet[int], float, float], float] \
                    = weak_shapley_score,
                 min_atoms: int = 5,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize atoms explainer

        Parameters
        ----------
        predictor : Callable[[List[Mol]], List[float]]
            Converter of a list of molecules into a list of target property values 
            for the GCNN model.
        shapley_score: Callable[[FrozenSet[int], float, float], float]

        device : torch.device
            Device on which a torch.Tensor is or will be allocated.
            Default: cpu
        mode : str
            Type of GCNN: either classification or regression. 
            Default: classification
        min_atoms : int
            Number of atoms to stop analysing of sub-molecules
        """

        self.predictor = predictor
        self.shapley_score = shapley_score
        self.min_atoms = min_atoms
        self.device = device

    def __call__(self, mol: Mol, is_sym: bool = False) -> List[float]:
        """
        Call atoms explainer for a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecular whose atoms need to be explained.
        is_sym : bool
            Need to apply the molecule's symmetry.

        Returns
        -------
        atoms_scores : List[float]
            List of scores for the atoms of the molecular graph.
        """
        # Create atoms equivalence relation based on the molecule's symmetry
        if is_sym:
            sym_atoms = find_mol_sym_atoms(mol)
            sym_atoms_relation = {(x, y) for cls in sym_atoms for x in cls for y in cls}
            atom_relation = lambda x, y: (x, y) in sym_atoms_relation
        else:
            atom_relation = lambda x, y: x == y

        # Create the factor graph for input molecule. 
        # The nodes of factor graph are frozensets of equivalent atoms indexes.
        graph = to_graph(mol)
        factor_graph = nx.quotient_graph(graph, atom_relation)

        # Coalition is a list of nodes in the factor graph of input molecule.
        # The coalition's ID is a frozenset of all atoms indexes of all its nodes.
        coalition = list(factor_graph.nodes())
        key = hash(coalition_id(coalition))
        stack = [ key ]
        attrs = {  key : (coalition, atom_relation, factor_graph) }

        mols = [ mol ] # list of submolecules forming a batch for predicting.
        keys = [ key ] # list of keys of coalitions whose submolecules 
                       # forming a batch for predicting.
        processing_queue = [] # list of triples (n, parent_key, child_key), where n is
                              # a node in factor_graph, parent_key is a key of parent
                              # coalition, and child_key is a key of child coalition
                              # obtained from parent coalition by removing node n.

        # Main loop for all coalitions in the factor graph of input molecule.
        while len(stack) > 0:
            cur_key = stack.pop(0)
            cur_coalition, cur_atom_relation, cur_factor_graph = attrs[cur_key]

            # Iterate all nodes in the current coalition
            for node in cur_coalition:

                # Remove node from the current coalition
                child_coalition = [n for n in cur_coalition if n != node]
                child_subgraph = cur_factor_graph.subgraph(child_coalition)

                # Skip nodes, the removal of which makes the factor graph disconnected.
                # They will be processed later when smaller coalitions containing 
                # this node are processed.
                if not nx.is_connected(child_subgraph):
                    continue

                # Skip small coalitions
                child_coalition_id = coalition_id(child_coalition)
                if len(child_coalition_id) < self.min_atoms:
                    continue

                # We need to add node to the processing queue in order to take into 
                # account the contribution of node in the parent's coalition prediction
                # relative to the child's coalition prediction.
                child_key = hash(child_coalition_id)
                processing_queue.append((node, cur_key, child_key))

                # Child's coalition has already been added to the stack
                if child_key in attrs:
                    continue

                # Store the child's submolecule for the subsequent calculation 
                # of its prediction.
                child_mol = submolecule(mol, child_coalition_id)
                mols.append(child_mol)
                keys.append(child_key)

                child_atom_relation = cur_atom_relation
                child_factor_graph = cur_factor_graph

                # If necessary, add the child's coalition to stack
                if len(child_coalition_id) > self.min_atoms:

                    # If necessary, recalculate atoms equivalence relation based on 
                    # the child's submolecule symmetry. We need to recalculate
                    # the factor graph, since the child's submolecule can contain 
                    # equivalent atoms that are not equivalent in the input molecule.
                    if is_sym:
                        sym_atoms = find_mol_sym_atoms(child_mol)
                        non_trivial_clss = [cls for cls in sym_atoms if len(cls) > 1]
                        if non_trivial_clss:
                            new_to_old_atom_idx = {
                                i : x for i, x in enumerate(sorted(child_coalition_id))
                            }
                            sym_atoms_relation = {
                                (new_to_old_atom_idx[i], new_to_old_atom_idx[j]) 
                                for cls in non_trivial_clss for i in cls for j in cls
                            }
                            child_atom_relation = \
                                lambda x, y: x == y or (x, y) in sym_atoms_relation
                            child_factor_graph = nx.quotient_graph(graph, 
                                                                   child_atom_relation)
                            atom_to_new_node = {
                                i : n for n in child_factor_graph.nodes() for i in n
                            }
                            new_node_to_old_node = {
                                atom_to_new_node[i] : n for n in child_coalition for i in n
                            }
                            child_coalition = list(new_node_to_old_node.keys())

                    if len(child_coalition) > 1:
                        stack.append(child_key)

                # Store the child's coalition attributes
                attrs[child_key] = (child_coalition, 
                                    child_atom_relation, 
                                    child_factor_graph)

        # Calculate predictions for submolecules
        predictions = { keys[i]: p for i, p in enumerate(self.predictor(mols)) }

        atoms_scores = torch.zeros(mol.GetNumAtoms(),
                                   dtype=torch.float32, device=self.device)

        # Calculate scores of atoms in the input molecule.
        # The score of an atom A is the total relative contribution of all processed 
        # nodes N of the factor graph containing A into the parent's coalition prediction 
        # of N relative to the child's coalition prediction of N. All possible 
        # parent's and child's coalitionы of node N were determined at the previous stage.
        for node, parent_key, child_key in processing_queue:
            parent_prediction = predictions[parent_key]
            child_prediction = predictions[child_key]
            atoms_scores[list(node)] += self.shapley_score(node, 
                                                           parent_prediction, 
                                                           child_prediction)

        return atoms_scores.tolist()