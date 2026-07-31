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


---


## Lo primero: este documento tiene "Amenazas a la Validez"

Y flagea tú mismo el problema de la semilla, el confound del LR, y la limitación de CIFAR. Después de diez documentos, la trayectoria metodológica es visible y va en la dirección correcta.

Ahora vamos al 48%, porque creo que sé lo que es.

---

## El "atractor" es probablemente una constante de la distribución, no un fenómeno

Si tu $N_{eff}$ es el ratio de participación estándar:

$$N_{eff} = \frac{\left(\sum_i |g_i|\right)^2}{\sum_i g_i^2} \quad\Rightarrow\quad \frac{N_{eff}}{D} = \frac{\mathbb{E}[|g|]^2}{\mathbb{E}[g^2]}$$

**Ese cociente no mide dispersión. Mide la forma de la distribución.** Y tiene valores cerrados:

| Distribución de $g$ | $N_{eff}/D$ |
|---|---|
| Uniforme $[0,1]$ | 0.750 |
| **Gaussiana** | $2/\pi = $ **0.6366** |
| Exponencial | 0.500 |
| **Producto de dos gaussianas** $XY$ | $4/\pi^2 = $ **0.4053** |

Ahora tu barrido:

| $D$ | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| ratio | **0.632** | 0.563 | 0.531 | 0.480 | **0.457** |

$D{=}512$ da **0.632** contra $2/\pi = 0.6366$. **Menos de un 1% de diferencia.** Y la serie decae monótonamente hacia ~0.4, que es $4/\pi^2$.

Y hay una razón mecánica para que aparezca justo el producto de dos gaussianas. Con $y = W_2(g\odot h)$:

$$\frac{\partial L}{\partial g_i} = \underbrace{(\delta^\top W_2)_i}_{\approx\,\mathcal{N}} \cdot \underbrace{h_i}_{\approx\,\mathcal{N}}$$

**El gradiente del gate es un producto de dos cantidades aproximadamente gaussianas e independientes.** Su ratio de participación es exactamente $4/\pi^2$.

Hipótesis: no hay atractor emergente. Los gates heredan la forma de su primer gradiente, y las desviaciones a $D$ pequeño vienen de que con pocas features cada gate recibe más señal y la optimización los "gaussianiza".

**Test, cinco minutos:** mide $N_{eff}/D$ **después de un solo paso desde init=0**. Si ya está en ~0.45, no hay fenómeno que explicar.

Y el segundo, que separa las dos hipótesis de raíz: **entrena con etiquetas aleatorias.** Si el ratio sigue saliendo ~0.48, no tiene nada que ver con la tarea.

---

## Tu propia tabla contradice tu conclusión 1

Escribes que el nivel de dispersión es *"una propiedad intrínseca de la complejidad de la tarea"*.

| Tarea | Accuracy | ratio |
|---|---|---|
| MNIST 3 capas | 96.48% | ~0.55 |
| Fashion-MNIST | 85.32% | 0.513 |
| CIFAR-10 | 42.99% | 0.491 |

**Tres tareas con 53 puntos de diferencia en dificultad dan el mismo ratio.** Eso es evidencia directa de que el ratio **no** depende de la tarea. Depende de $D$ y del sustrato — que es justo lo que muestra tu barrido de escalado.

La conclusión correcta es más limpia y la tienes medida: *el ratio es insensible a la tarea y sensible a $D$*.

---

## Un resultado gratis que está en tus datos y no has visto

| | Params entrenables | Accuracy |
|---|---|---|
| 3 capas, $D{=}4096$ | $4096+4096+10 = $ **8.202** | **96.48%** |
| 2 capas, $D{=}8192$ | $8192+10 = $ **8.202** | 95.57% |

**Exactamente el mismo presupuesto. +0.91% a favor de la profundidad.**

Tu conclusión 3 ("depth gain") comparaba 96.48% contra 94.27%, que tiene el doble de parámetros y es un confound. Pero la comparación limpia está en la tabla de al lado y sale a tu favor igualmente. Es tu mejor resultado del documento y está sin enunciar.

*(Y el baseline que falta: regresión logística sobre píxeles crudos son 7.850 params y da ~92.5% en MNIST. Le ganas con 8.202. Eso es lo que hace que el número signifique algo.)*

---

## Lo que PR no te dice, y es la pregunta que importa

$N_{eff}$ es un estadístico de forma. **No te dice que puedas prescindir del 52% de los gates.**

El experimento que convierte esto en algo accionable:

> Ordena los gates por $|g|$. Pon a cero los $D - N_{eff}$ menores. Mide accuracy.

- Si aguanta → la oligarquía es real, tienes poda estructurada, y el nombre está justificado.
- Si se hunde → PR era una descripción de la distribución y nada más.

Y la curva completa (accuracy vs fracción podada, contra poda aleatoria como control) es el resultado que buscas. Barato, y decide si toda la línea tiene consecuencias.

---

## Sobre el escalado sublineal: predicción que discrimina

$\alpha \approx 0.88$. Dos hipótesis incompatibles:

**H-A: el ratio converge a $4/\pi^2$.** Entonces $\alpha \to 1$ y la sublinealidad es un transitorio de $D$ pequeño.

**H-B: $N_{eff}$ satura.** Entonces estás midiendo la **dimensión efectiva del kernel de random features** para esa tarea — que sí es una cantidad con significado, porque el kernel converge a un límite fijo cuando $D\to\infty$ y su espectro deja de crecer.

**$D = 16384, 32768$ las separa.** Si satura, tienes un estimador barato de la dimensión intrínseca de una tarea bajo un mapa de features. Eso es medible, interpretable, y sería un resultado de verdad.

*(Y si satura, debería saturar en **valores distintos** para MNIST y CIFAR. Ahora mismo no lo hacen, lo cual apunta a H-A.)*

---

## Literatura

| | |
|---|---|
| 🔴 **Frankle, Schwab & Morcos (2021), *Training BatchNorm and Only BatchNorm*** | Entrenar solo $\gamma,\beta$ por canal en una red congelada. **Es tu experimento con convs.** Y analizan la distribución de $\gamma$ aprendida. Lectura obligatoria. |
| 🔴 **Ramanujan et al. (2020), *What's Hidden in a Randomly Weighted Neural Network?*** | Máscaras binarias sobre pesos aleatorios (edge-popup). **Encuentran que conservar ~50% es lo óptimo.** Tu número aparece ahí. |
| Zhou et al. (2019), *Deconstructing Lottery Tickets* | Supermasks. |
| Malach et al. (2020) | Teorema del lottery ticket fuerte: una red aleatoria suficientemente ancha *contiene* la subred. Es la justificación teórica de tu premisa. |
| **Extreme Learning Machines** (Huang et al., 2006) | Capa oculta aleatoria, entrenar solo la salida. Literatura enorme, décadas. |
| Rahimi & Recht (2007) | Random features, dimensión efectiva. |
| Gaier & Ha (2019), *Weight Agnostic NNs* | La arquitectura importa más que los pesos. |

Que el 50% aparezca en Ramanujan et al. de forma independiente es un dato: o hay algo real, o hay una constante distribucional que se cuela en dos formalizaciones distintas de lo mismo. Mi apuesta es la segunda, pero merece comprobarse.

---

## Qué correr

1. **$N_{eff}$ tras un paso.** Cinco minutos. Puede cerrar la pregunta entera.
2. **Etiquetas aleatorias.** Separa "tarea" de "sustrato".
3. **Curva de poda real** vs poda aleatoria. Convierte PR en algo accionable.
4. **$D = 16384, 32768$.** Discrimina saturación frente a $4/\pi^2$.
5. Enuncia el resultado de profundidad a 8.202 params igualados, que ya lo tienes.

Y quita "Oligarquía". El nombre presupone la conclusión —que unos pocos gates dominan— que es exactamente lo que PR no demuestra. Llámalo "ratio de participación de gates en sustratos congelados" hasta que la curva de poda diga otra cosa.