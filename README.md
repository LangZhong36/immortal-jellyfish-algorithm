<div align="center">

# Immortal Jellyfish Algorithm (IJA)

**A novel bio-inspired swarm intelligence optimizer**  
*Modelling the immortal jellyfish lifecycle × lighthouse light propagation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Java 8+](https://img.shields.io/badge/Java-8%2B-ED8B00?logo=openjdk&logoColor=white)](https://www.java.com/)
[![Stars](https://img.shields.io/github/stars/LangZhong36/immortal-jellyfish-algorithm?style=social)](https://github.com/LangZhong36/immortal-jellyfish-algorithm)

</div>

---

## What is IJA?

The **Immortal Jellyfish Algorithm (IJA)** is a metaheuristic optimizer
inspired by two elegant natural phenomena:

1. **The lifecycle of *Turritopsis dohrnii*** — the only known animal that
   can revert from adult medusa to juvenile polyp form, cycling through
   four life stages indefinitely.
2. **Lighthouse light propagation** — the inverse-square law governing
   how light intensity falls off with distance from a point source.

Available in **Python**, **C++**, and **Java**.

---

## Four-Phase Lifecycle

```
 ──────────────────────────────────────────────────────────────▶  τ = t / T_max
  [0, 0.15)               [0.15, 0.50)        [0.50, 0.85)        [0.85, 1]
 ┌──────────────┐       ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │   POLYP      │   ─>  │ STROBILATION │ ─> │    MEDUSA    │ ─> │  SENESCENCE  │
 │              │       │              │    │              │    │              │
 │ Lévy-flight  │       │ Adaptive     │    │ Lighthouse   │    │ Gaussian     │
 │ exploration  │       │ ephyra buds  │    │ attraction   │    │ refinement   │
 └──────────────┘       └──────────────┘    └──────────────┘    └──────────────┘
   Global search           Diversify           Converge            Fine-tune
```

### Key Mathematical Operators

**Lighthouse Attraction** (Medusa phase):

$$I_{i,s} = \frac{I_0}{1 + \kappa\|\mathbf{x}_i - \mathbf{g}_s\|^2} \cdot e^{-\lambda\tau}$$

**Lévy-Flight Exploration** (Polyp phase):

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + 0.01(u-l) \cdot L_j(\beta) \cdot (\mathbf{x}_i^t - \mathbf{x}^*)$$

**Pulsed Swimming** (Medusa phase):

$$\delta_j^t = A_0(1-\tau)^2 \cdot \sin(2\pi f_p \tau + \phi_i) \cdot (u_j - l_j)$$

**Transdifferentiation** (opposition-based restart):

$$\mathbf{x}_i^{\text{opp}} = l_j + u_j - \mathbf{x}_i$$

Full derivations: [`docs/algorithm_details.md`](docs/algorithm_details.md)

---

## Benchmark Results

![Convergence Curves](results/convergence_curves.png)

![Rank Heatmap](results/rank_heatmap.png)

| Function | IJA | PSO | DE | GWO | WOA | SCA |
|:---------|:---:|:---:|:--:|:---:|:---:|:---:|
| Sphere | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Rosenbrock | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Rastrigin | **1st** | 4th | 2nd | 3rd | 5th | 6th |
| Ackley | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Griewank | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Levy | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Schwefel | **1st** | 3rd | 2nd | 4th | 5th | 6th |
| Zakharov | **1st** | 3rd | 2nd | 4th | 5th | 6th |

> Run `python python/experiments/run_all.py` then `python python/experiments/visualize.py` to reproduce.

---

## Quick Start

### Python (Primary Implementation)

```bash
pip install -r requirements.txt

# Minimise the Sphere function in 30 dimensions
python - <<'EOF'
import sys; sys.path.insert(0, "python")
from ija import IJA
import numpy as np

result = IJA(n=50, max_iter=500, seed=0).optimize(
    lambda x: np.sum(x**2), dim=30, lb=-100.0, ub=100.0
)
print(f"Best fitness : {result.best_fitness:.6e}")
print(f"NFev         : {result.n_function_evals}")
EOF
```

More examples: `python python/examples/quickstart.py`

### C++

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

./ija_demo                     # runs all 8 benchmarks

echo "1
3 30 500 -5.12 5.12" | ./ija_main   # interactive: fid d maxIter lb ub
```

### Java

```bash
cd java
mvn compile
mvn exec:java -Dexec.mainClass="ija.Main"

# Or without Maven:
javac -d out src/main/java/ija/*.java
java -cp out ija.Main
```

---

## Project Structure

```
immortal-jellyfish-algorithm/
│
├── python/                        ← PRIMARY IMPLEMENTATION
│   ├── ija/
│   │   ├── __init__.py            # Public API: IJA, OptimizeResult
│   │   ├── algorithm.py           # Main optimizer class
│   │   ├── operators.py           # All mathematical operators
│   │   ├── lifecycle.py           # Phase scheduler & transition functions
│   │   └── archive.py             # Crowding-distance elite archive
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── functions.py           # 8 benchmark functions + BenchmarkSpec
│   ├── compare/
│   │   ├── __init__.py
│   │   └── algorithms.py          # PSO, DE, GWO, WOA, SCA baselines
│   ├── experiments/
│   │   ├── run_all.py             # 30 runs × 6 algos × 8 funcs → JSON
│   │   └── visualize.py           # 7 figure types (convergence, boxplot, …)
│   └── examples/
│       └── quickstart.py          # 4 annotated usage examples
│
├── cpp/                           ← C++ (competitive-style)
│   ├── include/ija.h
│   ├── src/
│   │   ├── ija.cpp
│   │   ├── main.cpp
│   │   └── benchmark_demo.cpp
│   └── CMakeLists.txt
│
├── java/                          ← Java (standard style)
│   ├── src/main/java/ija/
│   │   ├── IJA.java               # Builder pattern, full algorithm
│   │   ├── OptimizeResult.java
│   │   ├── EliteArchive.java
│   │   ├── BenchmarkFunctions.java
│   │   └── Main.java
│   └── pom.xml
│
├── docs/
│   └── algorithm_details.md       # Full mathematical specification
├── results/                       # Figures & JSON (generated, gitignored)
│
├── pyproject.toml                 # pip-installable Python package
├── requirements.txt
├── CMakeLists.txt                 # (root redirect; build inside cpp/)
├── LICENSE
└── README.md
```

---

## IJA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 50 | Population size |
| `max_iter` | 500 | Maximum iterations |
| `top_k` | 3 | Number of lighthouse elites |
| `levy_beta` | 1.5 | Lévy exponent |
| `archive_size` | 15 | Elite archive capacity |
| `age_max` | 25 | Stagnation → transdifferentiation |
| `decay_lambda` | 0.5 | Lighthouse intensity decay rate |
| `seed` | None | Random seed for reproducibility |

---

## Algorithmic Improvements (IJA vs original LJA)

- **Adaptive ephyra count** — starts at 5, decreases to 2 through strobilation phase; larger early diversity, lower late cost.
- **Crowding-distance archive** — prunes archive by (fitness, crowding-distance) to retain diverse elites.
- **Senescence Gaussian refinement** — dedicated tight local search in final phase for higher precision.
- **Modular codebase** — operators, lifecycle scheduler, and archive are fully independent modules.

---

## Citation

```bibtex
@software{ija2025,
  title   = {Immortal Jellyfish Algorithm (IJA)},
  author  = {IJA Contributors},
  year    = {2025},
  url     = {https://github.com/LangZhong36/immortal-jellyfish-algorithm},
  license = {MIT}
}
```

---

## Contributing

1. Fork → feature branch (`git checkout -b feature/my-improvement`)
2. Commit (`git commit -m 'Add ...'`)
3. Push → Pull Request

All three language implementations should remain feature-equivalent.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

*"Like T. dohrnii transcends mortality, IJA transcends local optima."*

⭐ Star this repo if you find it useful!

</div>
