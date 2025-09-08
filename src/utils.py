# External imports
from rdkit.Chem.rdchem import Mol
import torch
from torch.nn.functional import softmax
from torch_geometric.data import Batch, Data
from typing import Callable, List, FrozenSet

# Internal imports
from methods import AtomsExplainer


def GnnNetsGR2valueFunc(gnnNets, target):
    def value_func(batch) -> List[float]:
        with torch.no_grad():
            result = gnnNets(batch)
            score = result[:, target]
        return score.tolist()

    return value_func


def GnnNetsGC2valueFunc(gnnNets, target):
    def value_func(batch) -> List[float]:
        with torch.no_grad():
            logits = gnnNets(batch)
            probs = softmax(logits, dim=-1)
            score = probs[:, target]
        return score.tolist()
    return value_func


def get_scores(mol : Mol, 
               featurizer : Callable[[Mol], Data], 
               explainable_model : torch.nn.Module, 
               target : int,
               explainer_kwargs,
               mode : str = "classification", 
               device : torch.device = torch.device("cpu"),
               is_sym : bool = False):
    
    if mode == "classification":
        predict_func = GnnNetsGC2valueFunc(explainable_model, target)
    elif mode == "regression":
        predict_func = GnnNetsGR2valueFunc(explainable_model, target)
    else:
        predict_func = GnnNetsGC2valueFunc(explainable_model, target)

    def predictor(mols: List[Mol]) -> List[float]:
        data = Batch.from_data_list([ featurizer(mol) for mol in mols ])
        return predict_func(data)

    explainer = AtomsExplainer(predictor, device=device, **explainer_kwargs)

    scores = explainer(mol, is_sym=is_sym)

    return scores


def coalition_id(coalition: List[FrozenSet[int]]) -> FrozenSet[int]:
    return frozenset().union(*coalition)


def weak_shapley_score(node : FrozenSet[int],
                       parent_prediction : float, 
                       child_prediction : float) -> float:
    return (1 - child_prediction / parent_prediction) / len(node)
