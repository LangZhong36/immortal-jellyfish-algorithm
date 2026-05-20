<div align="center">

# 🪼 灯塔水母算法（IJA）

**新型生物启发式群体智能优化器**  
*融合灯塔水母生命周期 × 灯塔光传播物理模型*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/)
[![Java 8+](https://img.shields.io/badge/Java-8%2B-ED8B00?logo=openjdk&logoColor=white)](https://www.java.com/)
[![Stars](https://img.shields.io/github/stars/LangZhong36/immortal-jellyfish-algorithm?style=social)](https://github.com/LangZhong36/immortal-jellyfish-algorithm)

**[English README](README.md)**

</div>

---

## 🌊 什么是 IJA？

**灯塔水母算法（Immortal Jellyfish Algorithm，IJA）** 是一种全新的元启发式优化算法，灵感来源于：

1. ***Turritopsis dohrnii*（灯塔水母）的生命周期** —— 地球上唯一已知能从成体水母体退化回幼体水螅体、实现生物学意义上"永生"的动物。
2. **灯塔水母光传播的物理模型** —— 光强随距离按平方反比衰减的物理规律。

提供 **Python**、**C++**、**Java** 三个版本。

---

## 🔬 四阶段生命周期

```
 ──────────────────────────────────────────────────────────────▶  进度 τ = t/T
  [0, 0.15)              [0.15, 0.50)        [0.50, 0.85)         [0.85, 1]
 ┌──────────────┐      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │   水螅体      │  ->  │  横裂生殖    │ -> │   水母体      │ -> │    衰老期    │
 │   POLYP      │      │ STROBILATION │    │    MEDUSA    │    │  SENESCENCE  │
 │              │      │              │    │              │    │              │
 │  莱维飞行     │      │ 自适应       │    │  灯塔吸引     │    │  高斯精细     │
 │  全局探索     │      │ 子代克隆     │    │  开发收敛     │    │  局部优化     │
 └──────────────┘      └──────────────┘    └──────────────┘    └──────────────┘
     广域搜索               多样化              收敛                精调
```

**核心数学算子：**

$$I_{i,s} = \frac{I_0}{1+\kappa\|\mathbf{x}_i-\mathbf{g}_s\|^2}\,e^{-\lambda\tau} \qquad \text{（灯塔吸引，逆平方光强模型）}$$

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + 0.01(u-l)\cdot L_j(\beta)\cdot(\mathbf{x}_i^t-\mathbf{x}^*) \qquad \text{（莱维飞行探索）}$$

$$\delta_j^t = A_0(1-\tau)^2\sin(2\pi f_p\tau+\phi_i)(u_j-l_j) \qquad \text{（脉冲游泳）}$$

$$\mathbf{x}_i^{\text{opp}} = l_j + u_j - \mathbf{x}_i \qquad \text{（去分化，基于对立学习的重启）}$$

完整数学推导见 [`docs/algorithm_details.md`](docs/algorithm_details.md)

---

## 📊 基准测试结果

实验设置：**30 次独立运行**，维度 **D = 30**。  
IJA 使用推荐参数；对比算法使用各自文献标准参数。

### 收敛曲线

![收敛曲线](results/convergence_curves.png)

### 排名热力图

![排名热力图](results/rank_heatmap.png)

### 平均排名柱状图

![平均排名](results/average_rank_bar.png)

### 最终适应度分布箱线图

![箱线图](results/boxplots.png)

---

## 🚀 快速使用

### Python

```bash
# 安装依赖
pip install -r requirements.txt

# 运行一个简单示例
python - <<'EOF'
import sys; sys.path.insert(0, "python")
from ija import IJA
import numpy as np

# 最小化 30 维球函数
result = IJA(n=100, max_iter=2000, seed=0).optimize(
    lambda x: np.sum(x**2),
    dim=30, lb=-100.0, ub=100.0
)
print(f"最优适应度：{result.best_fitness:.6e}")
print(f"函数评估次数：{result.n_function_evals}")
EOF
```

更多用法示例：`python python/examples/quickstart.py`

### C++

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

./ija_demo          # 运行所有 8 个基准函数
```

### Java

```bash
cd java
mvn compile
mvn exec:java -Dexec.mainClass="ija.Main"

# 不用 Maven：
javac -d out src/main/java/ija/*.java
java -cp out ija.Main
```

---

## 🔁 复现完整实验

```bash
# 第一步：运行实验（约 30-60 分钟，取决于机器性能）
cd immortal-jellyfish-algorithm
python python/experiments/run_all.py

# 第二步：生成所有图表
python python/experiments/visualize.py

# 生成全部图表但跳过 3D 轨迹图（节省时间）
python python/experiments/visualize.py --skip-slow
```

---

## 📁 项目结构

```
immortal-jellyfish-algorithm/
├── python/                        ← 主实现（Python）
│   ├── ija/
│   │   ├── __init__.py            # 对外 API：IJA, OptimizeResult
│   │   ├── algorithm.py           # 主优化器类
│   │   ├── operators.py           # 所有数学算子
│   │   ├── lifecycle.py           # 相位调度器
│   │   └── archive.py             # 拥挤距离精英档案
│   ├── benchmarks/functions.py    # 8 个基准函数
│   ├── compare/algorithms.py      # PSO、DE、GWO、WOA、SCA
│   ├── experiments/
│   │   ├── run_all.py             # 完整实验运行器
│   │   └── visualize.py           # 图表生成
│   └── examples/quickstart.py    # 4 个带注释的示例
├── cpp/                           # C++ 实现（竞赛风格）
│   ├── include/ija.h
│   ├── src/ija.cpp
│   └── CMakeLists.txt
├── java/                          # Java 实现（Builder 模式）
│   ├── src/main/java/ija/
│   └── pom.xml
├── docs/algorithm_details.md      # 完整数学推导
├── results/                       # 图表和 JSON 输出
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md                      # 英文主页
```

---

## ⚙️ IJA 推荐参数

| 参数 | 推荐值 | 说明 |
|------|:-----:|------|
| `n` | **100** | 种群规模 |
| `max_iter` | **2000** | 最大迭代次数 |
| `top_k` | 5 | 灯塔精英数量 |
| `levy_beta` | 1.5 | 莱维分布指数 |
| `archive_size` | 20 | 精英档案容量 |
| `age_max` | 20 | 停滞阈值（触发去分化） |
| `decay_lambda` | 0.3 | 灯塔光强衰减率 |
| `alpha_max` | 2.5 | 最大步长 |
| `alpha_min` | 0.01 | 最小步长 |

---

## 🆕 算法改进亮点

相比同类算法，IJA 的独特设计：

- **自适应子代数量**：横裂生殖阶段，ephyra 数量从 5 动态递减至 2，前期多样化、后期节省预算。
- **拥挤距离档案**：精英档案按（适应度, 拥挤距离）双重标准剪枝，保留高质量 + 多样性的精英解集。
- **四阶段自动调度**：无需手动切换策略，算法根据进度 τ 自动切换全局探索 → 多样化 → 收敛 → 精调。
- **三语言实现**：Python、C++、Java 功能对等，覆盖研究、嵌入式、工业多场景。

---

## 📖 引用格式

```bibtex
@software{ija2025,
  title   = {Immortal Jellyfish Algorithm (IJA): A Novel Bio-Inspired Swarm Intelligence Optimizer},
  author  = {IJA Contributors},
  year    = {2025},
  url     = {https://github.com/LangZhong36/immortal-jellyfish-algorithm},
  license = {MIT}
}
```

---

## 🤝 参与贡献

欢迎 Issue 和 PR！步骤：Fork → 新建分支 → 提交修改 → 发起 Pull Request。三个语言实现应保持功能一致。

---

## 📜 开源协议

MIT — 详见 [LICENSE](LICENSE)。

---

<div align="center">
⭐ 如果这个项目对你有帮助，欢迎点个 Star！


</div>
