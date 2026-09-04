# Crystal Graph Neural Network for Band Gap Prediction

A PyTorch graph neural network (GNN) that predicts the **electronic band gap of crystalline materials** from their atomic structure.

## Overview

Each material is represented as a graph:

* **Nodes:** atoms and their elemental properties
* **Edges:** neighboring atoms identified using `CrystalNN`
* **Edge features:** bond length and electronegativity difference
* **Global features:** crystal system

The model uses message passing between neighboring atoms, mean-pools the resulting node embeddings, and predicts the material's band gap in eV.

### Architecture

```text
Atomic Features (13) ──→ Node Projection (128)
                              │
Bond Features (2) ────→ Edge Projection (64)
                              │
                              ▼
                       5 GNN Layers
                              │
                              ▼
                        Mean Pooling
                              │
                              ▼
                     Graph Embedding (128)
                              │
                    + Crystal System (7)
                              │
                              ▼
                    MLP: 135 → 128 → 64 → 32 → 1
                              │
                              ▼
                       Band Gap (eV)
```

## Data

Materials are retrieved from the **Materials Project** using `MPRester`. Non-metallic materials with valid `CrystalNN` neighbor information are used.

Each atom is described by 13 features including:

* Atomic number
* Electronegativity
* Atomic mass
* Atomic radius
* Electron affinity
* Periodic table group/row
* Ionization energy
* Valence orbital and electron count

The target is the material's **band gap**.

## Training

* **Framework:** PyTorch
* **Optimizer:** Adam
* **Learning rate:** `0.0003`
* **Epochs:** `20`
* **Loss:** Mean Squared Error (MSE)
* **GNN layers:** `5`

Training loss is tracked and visualized using Matplotlib.

## Dependencies

```text
numpy
pandas
matplotlib
scipy
torch
pymatgen
mp-api
```

## Files

```text
├── GNN.py      # GNN architecture
├── main.py     # Data retrieval, feature generation, and training
└── README.md
```

## Future Work

* Add train/validation/test splits
* Increase dataset size
* Normalize input features
* Evaluate using MAE and RMSE
* Tune model architecture and hyperparameters
* Compare against other ML approaches
