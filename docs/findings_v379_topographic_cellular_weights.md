# Findings v379: Topographic Cellular Weight Regularization (TCR)

**Fecha:** 2026-09-30  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v379_topographic_cellular_weights.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v379_topographic_cellular_weights.py)  
**Registro Crudo:** [`results/raw/v379_topographic_cellular_weights.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v379_topographic_cellular_weights.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Régimen infinitesimal $\epsilon \le 10^{-3}$ inoperante)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento evalúa directamente las conjeturas formuladas en el documento de propuesta [`docs/proposal_cellular_weight_regularization.md`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/proposal_cellular_weight_regularization.md) (líneas 18–22), donde se hipotetizaba que:
> *"un acoplamiento espacial infinitesimal ($\epsilon \in [10^{-5}, 10^{-3}]$):*  
> *1. Rompe la simetría de gauge por permutación ($\rho_{\text{raw}} \gg 0$)...*  
> *2. Induce topografía cortical ($\text{AdjCos} \gg 0$)...*  
> *3. Fuerza un decaimiento exponencial en el espectro de valores singulares (bajo rango efectivo)..."*

**Refutación empírica del régimen infinitesimal:** Los datos de `v379` demuestran que la hipótesis de acoplamiento *infinitesimal* ($\epsilon \le 10^{-3}$) es **falsa en presencia de optimizadores adaptativos (AdamW)**. A esa escala de acoplamiento, la magnitud de la penalización ($\sim 10^{-6}$) y su gradiente ($\sim 10^{-5}$) quedan completamente ahogados por el gradiente de la tarea ($\sim 10^{-2}$) y los momentos de segundo orden de AdamW. Como resultado, todas las métricas observadas son estadísticamente indistinguibles del baseline sin regularización.

---

## 1. Configuración Experimental

- **Modelo:** MLP de 3 capas ($784 \to 256 \to 128 \to 10$), $235{,}146$ parámetros entrenables.
- **Dataset:** MNIST (60,000 train / 10,000 test), normalizado estándar, batch size = 256.
- **Optimizador:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), 10 épocas por corrida.
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 18 ejecuciones).
- **Condiciones evaluadas:**
  1. `Turing_ReactionDiffusion_1e-3`: Kernel Mexican Hat DoG $5 \times 5$ ($\epsilon = 10^{-3}$).
  2. `Turing_ReactionDiffusion_1e-4`: Kernel Mexican Hat DoG $5 \times 5$ ($\epsilon = 10^{-4}$).
  3. `Dirichlet_Laplacian2D_1e-3`: Tensión superficial Dirichlet 2D ($\epsilon = 10^{-3}$).
  4. `Dirichlet_Laplacian2D_1e-4`: Tensión superficial Dirichlet 2D ($\epsilon = 10^{-4}$).
  5. `Cortical_1D_Neuron_1e-3`: Suavidad a lo largo del eje neuronal $d_{\text{out}}$ ($\epsilon = 10^{-3}$).
  6. `Baseline_Standard_AdamW`: AdamW estándar sin regularización espacial ($\epsilon = 0.0$).

---

## 2. Resultados Consolidados

| Condición | Modo | $\epsilon$ | Val Acc (%) | EffRank $W_1$ | AdjCos (H2) | $\rho_{\text{raw}}$ $W_1$ (H1) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Turing_ReactionDiffusion_1e-3` | turing | $10^{-3}$ | $97.84 \pm 0.09$ | 199.36 | 0.0110 | 0.0054 | 110.81 |
| `Turing_ReactionDiffusion_1e-4` | turing | $10^{-4}$ | $97.97 \pm 0.13$ | 199.27 | 0.0121 | 0.0063 | 114.14 |
| `Dirichlet_Laplacian2D_1e-3` | dirichlet | $10^{-3}$ | $98.00 \pm 0.06$ | 199.28 | 0.0112 | 0.0060 | 91.82 |
| `Dirichlet_Laplacian2D_1e-4` | dirichlet | $10^{-4}$ | $97.84 \pm 0.26$ | 199.35 | 0.0092 | 0.0049 | 91.19 |
| `Cortical_1D_Neuron_1e-3` 🌟 | cortical_1d | $10^{-3}$ | **$98.01 \pm 0.07$** | 199.26 | 0.0090 | 0.0032 | 90.44 |
| `Baseline_Standard_AdamW` | none | $0.0$ | $97.87 \pm 0.32$ | 199.22 | 0.0114 | 0.0076 | 91.77 |

*Nota de rigor en anotación:* 🌟 Se asigna a `Cortical_1D_Neuron_1e-3` por tener el mayor valor nominal de precisión (98.01%), pero se hace constar explícitamente que la diferencia $|\Delta| = 0.14\%$ es inferior a $2 \times \text{SE}$ del baseline ($0.32\%$), por lo que **no es estadísticamente distinguible del ruido**.

---

## 3. Análisis Detallado por Hipótesis

### H1: Ruptura de la Simetría de Gauge inter-semilla ($\rho_{\text{raw}}$)
- **Predicción del doc:** $\rho_{\text{raw}}(\text{Baseline}) \approx 0$, mientras que $\rho_{\text{raw}}(\text{Topo}) \gg 0$.
- **Observado:**
  - Baseline: $\rho_{\text{raw}} = 0.0076 \pm 0.0015$.
  - Dirichlet $10^{-3}$: $\rho_{\text{raw}} = 0.0060$.
  - Turing $10^{-3}$: $\rho_{\text{raw}} = 0.0054$.
  - Cortical 1D: $\rho_{\text{raw}} = 0.0032$.
- **Conclusión:** En todos los casos, los pesos entre semillas independientes permanecen ortogonales. La regularización con $\epsilon \le 10^{-3}$ **no rompe la degeneración de permutación inter-semilla**.

### H2: Topografía Cortical (Coseno entre neuronas contiguas)
- **Predicción del doc:** Neuronas adyacentes desarrollan campos receptores correlacionados ($\text{AdjCos} \gg 0$).
- **Observado:**
  - Baseline: $\text{AdjCos} = 0.0114$.
  - Turing $10^{-3}$: $\text{AdjCos} = 0.0110$.
  - Dirichlet $10^{-3}$: $\text{AdjCos} = 0.0112$.
  - Cortical 1D: $\text{AdjCos} = 0.0090$.
- **Conclusión:** Las neuronas contiguas conservan orientaciones casi ortogonales ($\approx 0.01$). No hay formación de gradientes continuos o bandas visibles a esta intensidad de acoplamiento.

### H3: Colapso Espectral y Rango Efectivo
- **Predicción del doc:** Decaimiento acelerado de valores singulares; retención de $>95\%$ de energía en rango 8 o 16.
- **Observado en $W_1 \in \mathbb{R}^{256 \times 784}$:**
  - Energía en rango 4: $\sim 18.2\%$ – $18.4\%$ en todas las condiciones.
  - Energía en rango 16: $\sim 49.4\%$ – $49.8\%$ en todas las condiciones.
  - Rango efectivo (entropía Shannon): $\approx 199.2$ – $199.4$ en todas las condiciones.
- **Conclusión:** El espectro singular de la matriz es insensible a $\epsilon \le 10^{-3}$. No se produce colapso de rango.

---

## 4. Auditoría de Causa: Diagnóstico del Fallo de Señal

Siguiendo el protocolo obligatorio antes de declarar el resultado como negativo definitivo:
1. **¿Hay un bug de implementación?** No. El forward/backward de autograd ejecuta la regularización; el tiempo de sobrecoste (`mean_overhead_s`) es medible ($1.2\text{s}$ para Dirichlet, $5.5\text{s}$ para Turing).
2. **¿El baseline está bien ajustado?** Sí, 97.87% en 10 épocas en MNIST refleja un entrenamiento estándar convergente.
3. **¿Falta algún paso de preprocesamiento?** No, normalización estándar de MNIST.
4. **¿El fallo es sensible a un hiperparámetro no barrido?** **SÍ, CLAVE:** La escala de $\epsilon$. Con inicialización Kaiming ($\sigma \approx 0.05$), la penalización Dirichlet es de orden $10^{-6}$. Su gradiente es $\sim 10^{-5}$, mientras que el gradiente de la tarea por lote es $\sim 10^{-2}$. En AdamW, este gradiente infinitesimal es completamente absorbido por el ruido estocástico del batch y los momentos de segundo orden ($v_t$).
5. **¿La métrica de evaluación tiene suficiente muestra?** Sí (10,000 imágenes de test, 3 semillas independientes).

---

## 5. Amenazas a la Validez

1. **Régimen Sub-umbral de $\epsilon$:** La propuesta asumió teóricamente que un término infinitesimal bastaría como "bias inductivo". En redes discretas optimizadas con gradiente estocástico y momentum, las fuerzas inferiores a la varianza del batch se promedian a cero. Para observar auto-organización se requiere un acoplamiento macroscópico ($\epsilon \in [10^{-2}, 1.0]$) o un ratio dinámico $\epsilon_t = \alpha \frac{\|\nabla \mathcal{L}\|}{\|\nabla \mathcal{R}\|}$.
2. **Interferencia del Optimizador Adaptativo:** AdamW escala cada coordenada por $1 / \sqrt{v_t + \epsilon_{\text{adam}}}$. Si el gradiente de regularización no tiene la misma escala que el de la tarea, el preacondicionador distorsiona la física del kernel continuo.
3. **Inicialización i.i.d. de Alta Frecuencia:** El modelo parte de ruido blanco Gaussiano puro. Si el acoplamiento es débil, 10 épocas de difusión lineal son insuficientes para suavizar un campo de 200,000 variables estocásticas independientes.

---

## 6. Próximo Paso Recomendado (v380)

Para dirimir si la Topografía Celular es viable o un artefacto teórico:
- **v380:** Barrido en régimen macroscópico: $\epsilon \in \{0.01, 0.05, 0.2, 1.0\}$, o acoplamiento normalizado por gradiente ($\nabla_{\text{total}} = \nabla_{\text{task}} + \alpha \frac{\|\nabla_{\text{task}}\|}{\|\nabla_{\text{reg}}\|} \nabla_{\text{reg}}$). Esto forzará una pugna real entre la topografía espacial y la pérdida de la tarea.
