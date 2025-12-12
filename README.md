# Molecular Generation with Diffusion Models

*Generative AI for molecular design with target property control*

## Overview

This project implements diffusion models for de novo molecular generation, conditioned on target HOMO-LUMO gap values. The goal is to generate novel molecules with desired electronic properties for applications in materials design and drug discovery.

## Motivation

Traditional molecular discovery relies on expensive trial-and-error experimentation. Machine learning, particularly generative models, can accelerate this process by:
- Learning structure-property relationships from existing data
- Generating novel molecular candidates with target properties
- Reducing experimental screening time by 60%+ (based on my industry experience)

## Current Status

**Phase 1: Foundations** ✅ *Completed*
- Dataset preparation (QM9: 134k molecules)
- Property distribution analysis (HOMO, LUMO, energy gap)
- Cheminformatics toolkit setup (RDKit)
- Baseline exploratory data analysis

**Phase 2: Model Development** 🔄 *In Progress*
- Implementing forward diffusion process
- Building denoising neural network
- Training on molecular representations
- Conditional generation with property constraints

**Phase 3: Evaluation** 📅 *Planned*
- Validity, uniqueness, and novelty metrics
- Property prediction accuracy assessment
- Drug-likeness evaluation (Lipinski's Rule of Five)
- Comparison with baseline methods

## Technical Approach

### Architecture
- **Model:** Denoising Diffusion Probabilistic Models (DDPM)
- **Conditioning:** Target HOMO-LUMO gap values
- **Molecular Representation:** Extended Connectivity Fingerprints (ECFP) or Graph Neural Networks

### Dataset
- **Source:** QM9 (Quantum chemistry database)
- **Size:** 134,000 small organic molecules
- **Properties:** DFT-computed (B3LYP/6-31G(2df,p))
  - HOMO energy (eV)
  - LUMO energy (eV)
  - HOMO-LUMO gap (eV)
  - Additional quantum properties

### Technology Stack
- **ML Framework:** PyTorch
- **Cheminformatics:** RDKit
- **Molecular Modeling:** DFT background for validation
- **Infrastructure:** AWS (EC2, S3) for scalable training
- **Visualization:** Matplotlib, Seaborn

## Project Structure
```
molecular-diffusion-generation/
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 0-rdkit_basics.ipynb
│   └── qm9_exploration.ipynb
├── src/                    # Source code (to be added)
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   └── utils/             # Helper functions
├── models/                 # Saved model checkpoints
├── data/                   # Data files and visualizations
├── diffusion-notes/        # Research notes and references
└── README.md
```

## Installation
```bash
# Create conda environment
conda create -n molgen python=3.10
conda activate molgen

# Install core dependencies
pip install torch torchvision
pip install rdkit pandas matplotlib jupyter numpy

# Install molecular modeling tools
pip install torch-geometric
```

## Usage

*Coming soon: Training scripts and inference examples*

## Key Results

*Results will be added as model development progresses*

Expected outcomes:
- Generate 500+ novel molecules with target HOMO-LUMO gaps
- Achieve >80% validity rate (RDKit-parseable SMILES)
- Demonstrate property control within ±0.5 eV of target

## Scientific Background

### Why HOMO-LUMO Gap Matters
The HOMO-LUMO gap determines:
- **Reactivity:** Smaller gaps → more reactive molecules
- **Stability:** Larger gaps → more stable molecules
- **Optical properties:** Gap relates to absorption/emission wavelengths
- **Applications:** Battery materials, OLEDs, photovoltaics, drug design

### Diffusion Models for Molecules
Diffusion models offer advantages over GANs and VAEs:
- **Stable training:** No mode collapse
- **High sample quality:** State-of-the-art generation
- **Controllability:** Easy to condition on target properties
- **Interpretability:** Clear forward/reverse process

## References

**Diffusion Models:**
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models" [[arXiv](https://arxiv.org/abs/2006.11239)]
- Xu et al. (2022). "GeoDiff: Geometric Diffusion Model for Molecular Conformation" [[arXiv](https://arxiv.org/abs/2203.02923)]

**Dataset:**
- Ramakrishnan et al. (2014). "Quantum chemistry structures and properties of 134 kilo molecules" [[Nature](https://www.nature.com/articles/sdata201422)]

## Author

**Noushin Omidvar**  
ML Scientist | Chemical Engineering PhD  
[LinkedIn](https://linkedin.com/in/noushin-omidvar) | [Email](mailto:noush.omidvar@gmai.com)

*Bridging chemistry and AI to accelerate molecular discovery*

## License

MIT License - See LICENSE file for details

