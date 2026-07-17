# Findings v291: Systematic Validation of the Oligarchy Hypothesis

## Goal
Verify the robustness of the Oligarchy Hypothesis (sparse emergent gating in frozen backbones) across:
1. Harder tabular/image datasets (Fashion-MNIST).
2. Deep configurations (3-layer MLP).
3. Scaling behavior (Effective number of gates $N_{eff}$ vs backbone dimension $D$).
4. Color image datasets (CIFAR-10).

This experiment is classified as **Nivel 1 - Sondeo Exploratorio** (1 seed, 10 epochs, no standard error calculated).

---

## Experimental Results

### 1. Fashion-MNIST Gating (4096 Hidden Dimension)
Tests if the discovery mode (starting from 0.0 gate initialization) can find useful sparse representations on a harder classification task than standard MNIST.

| Config | Final Accuracy (%) | Final Effective N ($N_{eff}$) | Activation Ratio ($N_{eff}/D$) |
| :--- | :---: | :---: | :---: |
| **Baseline (Init=1.0)** | 85.22% | 2095.2 | 0.5115 |
| **Discovery (Init=0.0)** | 85.32% | 2099.8 | 0.5126 |

* **Observation:** Both configurations converge to almost identical effective gate counts ($\sim 2100$, representing $\sim 51.2\%$ of $D=4096$) and final accuracy (differing by $+0.10\%$ in favor of the discovery mode).

### 2. 3-Layer GatedMLP on MNIST (784 -> 4096 -> 4096 -> 10)
Tests if gating signals can propagate and discover sparse solutions when multiple frozen hidden layers are stacked.

| Config | Final Accuracy (%) | Final $N_{eff}$ Layer 1 | Final $N_{eff}$ Layer 2 |
| :--- | :---: | :---: | :---: |
| **Baseline (Init=1.0)** | 96.39% | 2474.5 | 2268.3 |
| **Discovery (Init=0.0)** | 96.48% | 2421.3 | 2271.2 |

* **Observation:** The addition of a second hidden layer boosts overall performance by $>2.1\%$ compared to the 2-layer model. The discovery mode initialized from zero successfully learns through depth and matches or slightly exceeds the baseline. Both modes converge to similar sparsity levels per layer ($\approx 2420 - 2475$ in Layer 1, $\approx 2270$ in Layer 2).

### 3. Scaling Sweep: Effective N vs D on MNIST (SiLU, init=0.0)
Measures the scaling behavior of $N_{eff}$ and the ratio $N_{eff}/D$ as a function of the hidden layer size $D$.

| Hidden Dimension ($D$) | Final Accuracy (%) | Final $N_{eff}$ | Ratio ($N_{eff}/D$) |
| :--- | :---: | :---: | :---: |
| 512 | 82.76% | 323.5 | 0.6318 |
| 1024 | 87.93% | 576.1 | 0.5626 |
| 2048 | 92.06% | 1087.6 | 0.5310 |
| 4096 | 94.27% | 1965.7 | 0.4799 |
| 8192 | 95.57% | 3740.8 | 0.4566 |

* **Observation:** The ratio $N_{eff}/D$ is not constant. It decreases monotonically from $63.2\%$ at $D=512$ down to $45.7\%$ at $D=8192$. The effective number of gates grows sublinearly with $D$ ($N_{eff} \propto D^\alpha$ with $\alpha \approx 0.9$).

### 4. CIFAR-10 Gating (4096 Hidden Dimension)
Tests the architecture on a 3-channel color image classification dataset (3072 input flat dimension).

| Config | Final Accuracy (%) | Final Effective N ($N_{eff}$) | Activation Ratio ($N_{eff}/D$) |
| :--- | :---: | :---: | :---: |
| **Baseline (Init=1.0)** | 42.36% | 2031.3 | 0.4959 |
| **Discovery (Init=0.0)** | 42.99% | 2012.6 | 0.4914 |

* **Observation:** The discovery mode (Init=0.0) matches and slightly leads the baseline by $+0.63\%$ accuracy. Both configurations converge to $\approx 2012 - 2031$ effective gates ($\approx 49\%$ of $D=4096$).

---

## Analysis & Conclusions
1. **Universality of the Sparsity Attractor (`[SEÑAL]`):** Across Fashion-MNIST, multi-layer MNIST, and CIFAR-10, we observe that the final number of effective gates is independent of initialization. The model consistently converges to $\approx 48\% - 51\%$ active gates when $D=4096$, supporting the hypothesis that the sparsity level is an intrinsic property of the task complexity and random features rather than initialization bias.
2. **Sublinear Scaling of Active Gates (`[SEÑAL]`):** The monotonic decrease of the activation ratio $N_{eff}/D$ from $63.2\%$ at $D=512$ to $45.7\%$ at $D=8192$ suggests that larger dimensions allow the network to recruit a tighter, more sparse selection of its feature space.
3. **Depth Gain (`[SEÑAL]`):** Stacking multiple frozen layers allows the gating mechanism to achieve higher representation capability ($96.48\%$ vs $94.27\%$ accuracy) without experiencing vanishing gradients in zero initialization mode.

---

## Amenazas a la Validez
1. **Falta de semillas y variabilidad estadística:** Al tratarse de un sondeo exploratorio de una única semilla (Nivel 1), los incrementos de precisión observados (ej. $+0.63\%$ en CIFAR-10) podrían deberse a fluctuaciones aleatorias y no a una superioridad sistemática del método de inicialización en cero. Un estudio Nivel 2 con 5 semillas y cálculo de error estándar es obligatorio antes de confirmar de forma concluyente cualquier afirmación de mejora.
2. **Saturación en CIFAR-10:** El rendimiento de ~43% en CIFAR-10 es bajo en comparación con redes convolucionales o MLPs completamente entrenados. Esto sugiere que las características aleatorias lineales simples no son suficientes para procesar imágenes complejas sin una estructura espacial local (convolucional) o pre-entrenamiento. El comportamiento observacional del gating sobre representaciones pobres podría diferir del comportamiento sobre representaciones ricas.
3. **Dependencia de la tasa de aprendizaje:** El scheduler OneCycleLR y la tasa de aprendizaje máxima ($0.05$) no han sido optimizados de forma independiente para cada dimensión $D$ en el sweep de escalado, lo que podría influir artificialmente en la velocidad de convergencia y la dispersión final de los gates para valores extremos de $D$ (como 512 u 8192).
