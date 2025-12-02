# Neural Local Wasserstein Regression

This repository contains code and experiments for learning regression maps between probability distributions using neural networks. We focus on a nonparametric framework that leverages the 2-Wasserstein distance and kernel-weighted loss to learn local linear transport maps between distributions.

Our method learns a family of conditional maps $T_{\mu_0}$ from source distributions $\mu_i$ to target distributions $\nu_i$, trained via a locally weighted loss:

$$
\hat{T}_{\mu_0} = \arg\min_{T(x) = \alpha + Bx} \sum_{i=1}^n W_2^2(T\#\mu_i, \nu_i) \cdot K_h(\mu_0, \mu_i)
$$

---

## 🧠 Key Features

* ✅ Distribution-on-distribution regression
* ✅ Supports empirical distributions and synthetic Gaussian mixtures
* ✅ Kernel-weighted Sinkhorn loss
* ✅ Locally affine maps via neural nets
* ✅ Modular, scriptable experiment framework
* ✅ Visualizations of learned pushforward maps
* ✅ Real data integration (MNIST)

---

## 🗂 Directory Structure

```
distributional_regression/
├── data/                      # Raw or preprocessed distributional data
│   └── raw/
│       ├── MNIST/            # Extracted from torchvision or preprocessed
│       └── generated/        # Synthetic Gaussian mixtures, etc.
│
├── results/                  # Experiment Results -- Validation losses, logs, metrics
│   ├── d2/                   # 2-Dim Gaussian Mixtures 
│   ├── figures/              # First wave -- Gaussians and early Gaussian Mixtures 
│   └── mnist/                # MNIST results
│
├── scripts/                  # High-level training/evaluation scripts
│   ├── train.py              # Main training script
│   ├── simulate.py           # simulate synthetic data
│   ├── eval_distance...      # IRRELEVANT
│   ├── preprocess_mnist.py   # Turn MNIST images to prob dists
│   ├── sweep_all_regimes.py  # Run all GMM experiment regimes
│   ├── sweep_regime.py       # Run one GMM experiment regimes
│   ├── visualize.py          # Visualize synthetic data (sanity check)
│   └── run_regime.py         # Generates metadata subset and trains model
│
├── src/                      # Core functionality
│   ├── estimator.py          # Model definition
│   ├── loss.py               # Sinkhorn, MSE, kernel loss
│   ├── local_fit.py          # Core local training logic
│   ├── data_utils/           # Data loading, sampling
│       ├── generator.py      # Generates synthetic data
│       ├── metadata.py       # Dataloader for GMM metadata (for experiment table)
│       ├── pair_dataset.py   # Dataloader 
│   └── utils/                # Kernel functions, metrics, plotting
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
conda create -n wasserstein python=3.9
conda activate wasserstein
pip install -r requirements.txt
```

### 2. Simulate data

```bash
python scripts/simulate.py --regime d2 --n_distributions 100 --n_samples 1000
```

### 3. Train a local model

```bash
python scripts/train.py --reference_id 42 --regime d2 --blur 0.01
```

### 4. Run full experiment sweep

```bash
python scripts/sweep_all_regimes.py
```

---

## 📊 Visualization

Figures are saved in `figures/phase*/` and include:

* Pushforward plots: source → predicted → target
* Loss curves
* Distance-vs-loss comparisons
* Heatmaps of kernel weights

---

## 🧪 Experimental Phases

| Phase       | Description                         |
| ----------- |-------------------------------------|
| Phase 1     | MSE baseline with no transport loss |
| Phase 2     | Sinkhorn divergence regression      |
| Phase 3     | Training visualization over epochs  |
| Phase 4     | Multiple reference measures         |
| Phase 5     | Blur and hyperparameter sweep       |
| Phase 6     | Coverage experiments                |
| Phase 7     | Normalized loss curve plots         |
| Phase 8     | Greedy reference selection          |
| Phase 9     | Fixed-k neighborhood tests          |
| Phase 10–12 | GMM data                            |

See [`figures/README.md`](results/figures/README.md) for details.

---

## 📦 Dependencies

* PyTorch
* GeomLoss
* NumPy, SciPy
* Matplotlib, Seaborn
* Scikit-learn
* tqdm

---

## ✏️ Citation / Acknowledgments

This work was developed as part of a PhD research project in applied mathematics focused on optimal transport and nonparametric regression. It draws inspiration from:

* [Cuturi, Marco. "Sinkhorn Distances." NeurIPS (2013)](https://arxiv.org/abs/1306.0895)
* [Okano et al. “Distribution-on-Distribution Regression with Wasserstein Metric.” JMLR (2022)](https://www.jmlr.org/papers/volume23/21-0247/21-0247.pdf)

---

## 📬 Contact

For questions, ideas, or collaboration, feel free to reach out.
