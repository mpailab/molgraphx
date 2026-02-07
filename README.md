# MolgraphX

A symmetry-sensitive method for interpreting the results of Graph Convolutional Neural Networks (GCNNs), which aims to explain the importance of individual atoms or fragments in a molecule for the GCNN predictions, while preserving molecular symmetry. This method provides a balance between formal accuracy and general chemical knowledge within a reasonable computational time.

## Installation

The package will be released to PyPI (pypi.org) soon:
```shell
pip install molgraphx
```
The package can be install from source using pip, for this you need to clone this repository, activate environment you want to add MolgraphX package to and install it using pip inside MolgraphX directory. Here conda env manager as example, any other (pip, poetry, uv, etc.) would work as well.
```shell
git clone https://github.com/mpailab/molgraphx.git
conda activate YOUR_PROJECT_ENV_NAME
pip install .
```
That would add molgraphx as package to you environment so you could use methods.AtomsExplainer for arbitrary model or use PyTorch model wrapper to calculate scores using utils.get_scores. 
All necessary packages listed in setup.py
```python
from rdkit import Chem
from torch_geometric.data import Data
import molgraphx.utils as U

# Dummy featurizer returning minimal Data objects
def featurizer(m):
    return Data()

class DummyModel(torch.nn.Module):
    def forward(self, batch):  # batch is ignored
        return torch.zeros((1, 2))  # logits -> softmax -> [0.5, 0.5]

mol = Chem.MolFromSmiles("CC")

scores = U.get_scores(
    mol,
    featurizer=featurizer,
    explainable_model=DummyModel(),
    target=0,
    explainer_kwargs={"min_atoms": mol.GetNumAtoms()},
    mode="classification",
)
```