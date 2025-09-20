# External imports
from rdkit.Chem.rdchem import Mol
import torch
from torch.nn.functional import softmax
from torch_geometric.data import Batch, Data
from typing import Callable, Iterable

# Internal imports
from molgraphx.methods import AtomsExplainer


def GnnNetsGR2valueFunc(gnnNets: torch.nn.Module, target: int):
    """Wrap a regression model to extract a target value from its output.

    Parameters
    ----------
    gnnNets : torch.nn.Module
        Model taking a batched graph object and returning predictions of shape
        ``[batch, D]`` or ``[batch]``.
    target : int
        Index of the target dimension to extract when the output is 2D.

    Returns
    -------
    Callable
        A function mapping a batched input to a 1D tensor of target scores.
    """
    def value_func(batch) -> torch.Tensor:
        with torch.no_grad():
            result = gnnNets(batch)
            score = result[:, target] if result.ndim == 2 else result
        return score

    return value_func


def GnnNetsGC2valueFunc(gnnNets: torch.nn.Module, target: int):
    """Wrap a classification model to output the probability of ``target``.

    Parameters
    ----------
    gnnNets : torch.nn.Module
        Model taking a batched graph object and returning class logits of shape
        ``[batch, C]``.
    target : int
        Class index whose probability to return.

    Returns
    -------
    Callable
        A function mapping a batched input to a 1D tensor of probabilities
        for the requested class. Applies ``softmax`` along the last dimension.
    """
    def value_func(batch) -> torch.Tensor:
        with torch.no_grad():
            logits = gnnNets(batch)
            probs = softmax(logits, dim=-1)
            score = probs[:, target]
        return score
    return value_func


def get_scores(
    mol: Mol,
    featurizer: Callable[[Mol], Data],
    explainable_model: torch.nn.Module,
    target: int,
    explainer_kwargs,
    mode: str = "classification",
) -> torch.Tensor:
    """Compute atom-level explanation scores for ``mol``.

    Parameters
    ----------
    mol : Mol
        Input RDKit molecule to explain.
    featurizer : Callable[[Mol], Data]
        Function that converts a single RDKit molecule to a PyG ``Data``.
    explainable_model : torch.nn.Module
        GNN model compatible with the provided featurizer.
    target : int
        Target index for regression output or class index for classification.
    explainer_kwargs : dict
        Keyword arguments to initialize ``AtomsExplainer``.
    mode : str
        Either ``"classification"`` or ``"regression"``.

    Returns
    -------
    torch.Tensor
        Per-atom scores aligned with the original atom indices of ``mol``.
    """

    if mode == "classification":
        predict_func = GnnNetsGC2valueFunc(explainable_model, target)
    elif mode == "regression":
        predict_func = GnnNetsGR2valueFunc(explainable_model, target)
    else:
        predict_func = GnnNetsGC2valueFunc(explainable_model, target)

    def predictor(mols: Iterable[Mol]) -> torch.Tensor:
        data = Batch.from_data_list([featurizer(m) for m in mols])
        return predict_func(data)

    explainer = AtomsExplainer(predictor, **explainer_kwargs)
    return explainer(mol)
