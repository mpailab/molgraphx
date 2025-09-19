# External imports
import rdkit
import torch
import networkx as nx
from rdkit.Chem.rdchem import Mol
from typing import Callable, List, Tuple, FrozenSet, Set, Dict
from collections import deque

# Internal imports
from molgraphx.molecule import find_mol_sym_atoms, submolecule, to_graph

# Type aliases
PredictorFn = Callable[[List[Mol] | Tuple[Mol]], List[float]]

SYMMETRY_MODE = set({'none', 'global', 'inner'})
SHAPLEY_MODES = set({'exclude', 'relative_exclude', 'synergy', 'coalition'})
    
class AtomsExplainer(object):
    """Explain atoms of molecular graphs"""

    def __init__(
        self,
        predictor: PredictorFn,
        min_atoms: int = 5,
        is_sym: bool = True,
        is_connect: bool = True,
        shapley_mode : str = 'exclude',
        symmetry_mode: str = 'inner',
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
        is_sym : bool
            Whether to taken into account molecular symmetry.
        is_connect : bool
            If False, allow coalitions of original atoms that induce a
            disconnected molecule (number of fragments > 1).
        shapley_mode : str
            Mode for calculating Shapley values of atoms. 
            Possible values:
            - 'exclude': ...
            - 'relative_exclude': ...
            - 'synergy': ...
            - 'coalition': ...
        symmetry_mode : str
            Mode to taken into account molecular symmetry.
            Possible values: 
            - 'none': does not use molecular symmetry;
            - 'global': use only symmetry of input molecule;
            - 'inner': use symmetry of submolecules;
        """

        self.predictor = predictor
        assert min_atoms > 0, \
               f"Bad value '{min_atoms}' for parameter min_atoms."
        self.min_atoms = min_atoms if min_atoms > 0 else 1
        self.is_sym = is_sym
        assert symmetry_mode in SYMMETRY_MODE, \
               f"Bad value '{symmetry_mode}' for parameter shapley_mode."
        self.symmetry_mode = symmetry_mode
        self.is_connect = is_connect
        assert shapley_mode in SHAPLEY_MODES, \
               f"Bad value '{shapley_mode}' for parameter shapley_mode."
        self.shapley_mode = shapley_mode
        self.device = device

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
        coalitions, mols = zip(*graph.nodes(data="molecule"))
        for i, p in enumerate(self.predictor(mols)):
            node = graph.nodes[coalitions[i]]
            if self.is_connect:
                node["prediction"] = p
            else:
                node["prediction"] = len(coalitions[i]) * (p / node["molecule"].GetNumAtoms())

        atom_scores = torch.zeros(atom_num, device=self.device, dtype=torch.float)

        if self.shapley_mode == 'exclude':

            atom_factor = torch.zeros(atom_num, device=self.device, dtype=torch.int)

            for u, v, atoms in graph.edges(data="atoms"):
                share = (graph.nodes[u]["prediction"] - graph.nodes[v]["prediction"]) / len(atoms)
                atom_scores[atoms] += share
                atom_factor[atoms] += 1

            atom_scores /= atom_factor

        elif self.shapley_mode == 'relative_exclude':

            for u, v, atoms in graph.edges(data="atoms"):
                share = (1 - graph.nodes[v]["prediction"] / graph.nodes[u]["prediction"]) / len(atoms)
                for i in atoms:
                    atom_scores[i] += share

        else:

            # Reverse topological order (bottom-up)
            topological_order = list(nx.topological_sort(graph))
            for coalition in reversed(topological_order):
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

                # # Distribute synergy uniformly over atoms of the current coalition
                # # (normalize by the number of atoms in the current submolecule)
                # share = node["synergy"] / len(coalition)
                # for i in coalition:
                #     atom_scores[i] += share

            if self.shapley_mode == 'synergy':

                for coalition, synergy in graph.nodes(data="synergy"):
                    share = synergy / len(coalition)
                    for i in coalition:
                        atom_scores[i] += share

            elif self.shapley_mode == 'coalition':

                # Topological order (top-bottom)
                for coalition in topological_order:
                    node = graph.nodes[coalition]

                    # Transitive closure of predecessors:
                    # union of direct parents and their predecessors
                    node["predecessors"] = set({coalition})
                    for pred_coalition in graph.predecessors(coalition):
                        pred = graph.nodes[pred_coalition]
                        node["predecessors"] |= pred["predecessors"]

                    if graph.out_degree[coalition] == 0:

                        synergy = 0
                        for pred_coalition in node["predecessors"]:
                            pred = graph.nodes[pred_coalition]
                            assert coalition <= pred_coalition
                            synergy += pred["synergy"] / len(pred_coalition)


                        for i in coalition:
                            # for n in graph.nodes:
                            #     preds = node["predecessors"]
                            #     assert not ((i in n) ^ (n in preds)), f"{i}, {n}\n{preds}"
                            atom_scores[i] += synergy
                
            else:
                raise ValueError("Bad value for parameter shapley_mode.")

        # for coalition in list(nx.topological_sort(graph)):
        #     node = graph.nodes[coalition]
        #     # print(coalition, node["prediction"], node["synergy"])

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
        - "orig_atoms": list[list[int]] mapping submolecule atom index -> list of
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

        if self.symmetry_mode == 'none':
            glob_sym_atoms = [ set({i}) for i in atom_ids ]
            glob_atom_relation = lambda x, y: x == y
            orig_atoms = [ [i] for i in atom_ids ]

        elif self.symmetry_mode == 'global':
            glob_sym_atoms = find_mol_sym_atoms(mol)
            glob_atom_relation = lambda x, y: x in glob_sym_atoms[y]
            orig_atoms = [ list(glob_sym_atoms[i]) for i in atom_ids ]

        elif self.symmetry_mode == 'inner':
            glob_sym_atoms = []
            glob_atom_relation = None
            orig_atoms = [ [i] for i in atom_ids ]

        else:
            raise ValueError("Bad value for parameter symmetry_mode.")

        graph = nx.DiGraph()
        graph.add_node(root, molecule=mol, orig_atoms=orig_atoms)

        while len(stack) > 0:
            coalition = stack.popleft()
            node_attrs = graph.nodes[coalition]
            cur_mol = node_attrs["molecule"]
            orig_atoms = node_attrs["orig_atoms"]

            # print("\n---------------------------------------------------")
            # print(coalition, cur_mol.GetNumAtoms())

            # Atom equivalence relation (symmetry-aware if requested)
            if self.symmetry_mode == 'inner':
                sym_atoms = find_mol_sym_atoms(cur_mol)
                atom_relation = lambda x, y: x in sym_atoms[y]
            else:
                sym_atoms = glob_sym_atoms
                atom_relation = glob_atom_relation

            # Factor (quotient) graph for the current submolecule
            cur_graph = to_graph(cur_mol)
            # print(cur_graph.nodes)
            fac_graph = nx.quotient_graph(cur_graph, atom_relation)

            # Articulation points: removing them disconnects the factor graph
            aps = set(nx.articulation_points(fac_graph))

            # Full set of atom indices for cur_mol (for quick difference)
            all_nodes_id = frozenset(range(cur_mol.GetNumAtoms()))

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
                child_coalition = frozenset({j for i in child_nodes_id 
                                               for j in orig_atoms[i]})

                # Already seen this coalition — skip duplicate
                if child_coalition in graph.nodes:
                    graph.add_edge(coalition, child_coalition,
                                   atoms=[j for i in node for j in orig_atoms[i]])
                    # print(coalition, " -> ", child_coalition)
                    continue

                # Build child submolecule and map back to original atoms
                atom_maps = []
                child_mol = submolecule(cur_mol, child_nodes_id, atom_maps,
                                        is_connect=self.is_connect)
                if child_mol.GetNumAtoms() < self.min_atoms:
                    continue

                # Map child atoms back to original atom sets.
                child_orig_atoms = [
                    [a for j in sym_atoms[atom_maps[i]] for a in orig_atoms[j]]
                    for i in range(child_mol.GetNumAtoms())
                ]

                # Register the submolecule node and enqueue it for further expansion
                graph.add_node(child_coalition, 
                               molecule=child_mol, 
                               orig_atoms=child_orig_atoms)
                graph.add_edge(coalition, child_coalition,
                               atoms=[j for i in node for j in orig_atoms[i]])
                stack.append(child_coalition)
                # print(coalition, " -> ", child_coalition)

        return graph

    # def _build_coalition_graph(self, mol: Mol) -> nx.DiGraph:
    #     """
    #     Build the coalition graph (DAG) for the given molecule.

    #     - Nodes: coalitions (frozenset of original atom indices). The induced
    #       subgraph can split into disconnected but equivalent components; we
    #       keep one representative connected component in the "molecule"
    #       attribute.
    #     - Edges: parent -> child if the child’s molecule is a submolecule of
    #       the parent’s (obtained by removing exactly one factor-graph node).

    #     Node attributes
    #     ---------------
    #     - "molecule": RDKit Mol of the representative connected submolecule.
    #     - "orig_atoms": list[list[int]] mapping submolecule atom index -> list of
    #       original atom indices in the input molecule.
    #     - "prediction": model output for this submolecule.
    #     - "synergy": contribution of this coalition to the model output 
    #       exclusive of all successors.
    #     - "successors": set of all nodes reachable from a given node.

    #     The root coalition contains all atom indices of the input molecule.
    #     """
    #     atom_ids = range(mol.GetNumAtoms())
    #     root = frozenset(atom_ids)
    #     stack = deque([root])

    #     if self.symmetry_mode == 'none':
    #         glob_sym_atoms = [ set({i}) for i in atom_ids ]
    #         glob_atom_relation = lambda x, y: x == y
    #         orig_atoms = [ [i] for i in atom_ids ]

    #     elif self.symmetry_mode == 'global':
    #         glob_sym_atoms = find_mol_sym_atoms(mol)
    #         glob_atom_relation = lambda x, y: x in glob_sym_atoms[y]
    #         orig_atoms = [ list(glob_sym_atoms[i]) for i in atom_ids ]

    #     elif self.symmetry_mode == 'inner':
    #         glob_sym_atoms = []
    #         glob_atom_relation = None
    #         orig_atoms = [ [i] for i in atom_ids ]

    #     else:
    #         raise ValueError("Bad value for parameter symmetry_mode.")

    #     graph = nx.DiGraph()
    #     graph.add_node(root, molecule=mol, orig_atoms=orig_atoms)

    #     while len(stack) > 0:
    #         coalition = stack.popleft()
    #         node_attrs = graph.nodes[coalition]
    #         cur_mol = node_attrs["molecule"]
    #         orig_atoms = node_attrs["orig_atoms"]

    #         # print("\n---------------------------------------------------")
    #         # print(coalition, cur_mol.GetNumAtoms())

    #         # Atom equivalence relation (symmetry-aware if requested)
    #         if self.symmetry_mode == 'inner':
    #             sym_atoms = find_mol_sym_atoms(cur_mol)
    #             atom_relation = lambda x, y: x in sym_atoms[y]
    #         else:
    #             sym_atoms = glob_sym_atoms
    #             atom_relation = glob_atom_relation

    #         # Factor (quotient) graph for the current submolecule
    #         cur_graph = to_graph(cur_mol)
    #         # print(cur_graph.nodes)
    #         fac_graph = nx.quotient_graph(cur_graph, atom_relation)

    #         # Articulation points: removing them disconnects the factor graph
    #         aps = set(nx.articulation_points(fac_graph))

    #         # Full set of atom indices for cur_mol (for quick difference)
    #         all_nodes_id = frozenset(range(cur_mol.GetNumAtoms()))

    #         # Iterate all nodes in the current factor graph
    #         for node in fac_graph.nodes():

    #             # print(node, node in aps, cur_mol.GetNumAtoms() < len(node) + self.min_atoms)

    #             # Create a copy of the graph to avoid modifying the original
    #             temp_graph = fac_graph.copy()
    #             temp_graph.remove_node(node)

    #             children = [ 
    #                 frozenset().union(*x) for x in nx.connected_components(temp_graph) 
    #             ]
    #             children_num = len(children)

    #             for child_nodes_id in children:

    #                 # Child coalition (original indexing)
    #                 child_coalition = frozenset({j for i in child_nodes_id 
    #                                                for j in orig_atoms[i]})

    #                 # Already seen this coalition — skip duplicate
    #                 if child_coalition in graph.nodes:
    #                     graph.add_edge(coalition, child_coalition,
    #                                    multiplicity = children_num,
    #                                    atoms=[j for i in node for j in orig_atoms[i]])
    #                     # print(coalition, " -> ", child_coalition)
    #                     continue

    #                 # Build child submolecule and map back to original atoms
    #                 atom_maps = []
    #                 child_mol = submolecule(cur_mol, child_nodes_id, atom_maps,
    #                                         is_connect=self.is_connect)

    #                 # Map child atoms back to original atom sets.
    #                 child_orig_atoms = [
    #                     [a for j in sym_atoms[atom_maps[i]] for a in orig_atoms[j]]
    #                     for i in range(child_mol.GetNumAtoms())
    #                 ]

    #                 # Register the submolecule node and enqueue it for further expansion
    #                 graph.add_node(child_coalition, 
    #                                molecule=child_mol, 
    #                                orig_atoms=child_orig_atoms)
    #                 graph.add_edge(coalition, child_coalition,
    #                                multiplicity = children_num,
    #                                atoms=[j for i in node for j in orig_atoms[i]])
                    
    #                 if child_mol.GetNumAtoms() >= self.min_atoms:
    #                     stack.append(child_coalition)

    #                 # print(coalition, " -> ", child_coalition)

    #     return graph
