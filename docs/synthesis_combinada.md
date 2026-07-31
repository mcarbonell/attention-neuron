# Síntesis Combinada: Portafolio de Investigación en Representaciones Compactas

> **Fuentes:** `research_portfolio_synthesis.md`, `experiment_catalog.md`, `research_synthesis_tools_not_hammers.md`
> **Autor:** Mario Carbonell
> **Período:** Enero 2026 – Julio 2026 (~7 meses)
> **Hardware:** AMD Ryzen 7 8845HS, Radeon 780M iGPU, 64 GB RAM + Colab + Modal.com
> **Experimentos catalogados:** 245 (v1–v299)

---

## 1. Tesis Central

**Hipótesis unificadora:** Las redes neuronales modernas desperdician grados de libertad aprendiendo conectividad sin estructura. Si reemplazamos matrices densas o estados opacos por generadores con priors geométricos, espectrales o de memoria, podemos obtener sistemas más pequeños, más interpretables y más fáciles de optimizar.

**Corolario operativo:** Una red eficiente debería reconocer qué estructura tiene el problema y seleccionar la representación, el operador, la memoria y el presupuesto de cómputo adecuados. Las matrices densas son un martillo universal: flexibles, pero caras y poco informativas.

---

## 2. Mapa de Herramientas por Régimen

| Régimen | Herramientas | Función |
|---|---|---|
| Estructura espacial local | CNN, Cone neurons, Haar | Localidad, bordes, invariancia |
| Estructura espectral | DCT, Walsh, Fourier | Compresión y mezcla estructurada |
| Periodicidad y oscilación | Resonancia, fase | Sintonía e interferencia |
| Interacciones multiplicativas | Espacio logarítmico, PROD | Convertir productos en operaciones simples |
| Discontinuidades | DGE, ramas simbólicas | Evitar gradientes engañosos |
| Recall asociativo | DeltaPhase holográfico | Memoria dependiente del contenido O(N) |
| Novedad y familiaridad | Atlas espectral, memoria de contraste | Saber cuándo el caso es conocido |
| Cómputo adaptativo | Surprise gate, MoE de competencia | Gastar recursos solo cuando hace falta |
| Hardware extremo | Ternarización, inhibición, GAP | Reducir multiplicaciones y memoria |
| Dinámica de entrenamiento | PID, stage gating | Separar exploración y refinamiento |

---

## 3. Las 7 Eras del Repositorio

| Era | Rango | Tema central | Hito representativo |
|---|---|---|---|
| 1. Fundacional | v1–v50 | Gating multiplicativo, neuronas geométricas (Stroke, Cone) | v18: 99% MNIST con gating; v50: Stroke Neurons 97.88% con 35K params |
| 2. Espectral | v51–v150 | Bases Walsh-Hadamard, Haar, DCT; sustitución de capas densas por síntesis espectral | v103–v106: Walsh reemplaza FFN denso; v125: Spectral Optimizer |
| 3. Holográfica | v151–v162 | Memoria asociativa por superposición de cristales, multiplexación, manifolds | v160: Compresión 937× con 83.45% en MNIST |
| 4. MoE Espectral | v163–v166 | Mixture of Experts espectral, especialización morfológica | v165: 93.27% con router + expertos de clase |
| 5. Auto-Arquitecto | v167–v170 | Neurogénesis, crecimiento orgánico de capas por corrección de errores | v170: 96.08% con 4 capas auto-ensambladas |
| 6. Resonancia | v171–v189 | Osciladores armónicos, votación por sintonía, clasificación sin capas densas | v186: Pure Resonance 96.12% con init determinista |
| 7. Fase y Regla Delta | v190–v299 | Memoria de fase compleja, Regla Delta, MQAR O(N), compresión espectral, optimizadores | v298: 99.95% MQAR en O(N); v299: ventaja compleja +22.84% |

---

## 4. Resultados Principales

### 4.1 Delta Phase — Memoria Asociativa O(N) [ANCLA]

**Progresión empírica en MQAR** (L=64, 8 pares clave-valor, d=64, 3 capas, ~110K params):

| Experimento | Mecanismo | MQAR Acc | Complejidad |
|---|---|---|---|
| v293 (Hebbiana pura) | M_t = M_{t-1} + K_t ⊗ V_t | 18.94% | O(N) |
| v296 (Mass Normalized) | + normalización causal | 23.59% | O(N) |
| v297 (Phase Softmax) | + forget gate + phase norm | 49.59% | O(N) |
| **v298 (Delta Rule)** | **+ corrección por error residual** | **99.95%** | **O(N)** |
| Softmax MHA (control) | softmax(QK^T/√d)V | 99.95% | O(N²) |

**Frontera de capacidad (v299)** — mismo presupuesto de memoria (~2,048 floats/cabeza):

| Representación | 8 pares (L=64) | 64 pares (L=512) |
|---|---|---|
| Compleja (Delta Phase) | 99.80% | **95.98%** |
| Real (DeltaNet Vanilla) | 99.67% | 73.14% |
| Softmax O(N²) (control) | 99.63% | 99.73% |

**Hipótesis mecanística:** La fase compleja hace más uniforme la geometría de las claves y reduce la diafonía causada por normas variables.

**Limitaciones:** Solo MQAR sintético (vocabulario N=32), secuencias L≤512, no integrado en LLM real, incluye Conv1D preprocesamiento.

### 4.2 Compresión Espectral con Reordenación de Canales [SEÑAL]

| Método | Ratio 90% (PPL) | Sin reordenar |
|---|---|---|
| Greedy TSP + Lowpass | **88.36** | 163.95 |
| Float32 original | 89.58 | 89.58 |

- Eliminar el 10% de coeficientes DCT de alta frecuencia **mejora** la perplejidad
- Zero-shot (sin dataset de calibración), a diferencia de GPTQ/AWQ
- La reordenación no altera la semántica del modelo (delta PPL < 10⁻⁵)

### 4.3 Gated Frozen Networks y la Hipótesis de la Oligarquía [SEÑAL]

| Propiedad | Observación |
|---|---|
| MNIST accuracy | 94.27% con 4,106 parámetros entrenables |
| Participation Ratio (N_eff/D) | ~48% consistentemente |
| Sensibilidad a init | Ninguna (init=0.0 ≈ init=1.0) |
| Weight decay | Destructivo |
| Fashion-MNIST | 85.32% |
| CIFAR-10 | 42.99% |

### 4.4 Neuronas Geométricas [SEÑAL]

- **Stroke Neurons (v50):** Curvas de Bézier entrenables (8 params/neurona). 97.88% MNIST con 35K params totales.
- **Cone Neurons (v101):** Cono 2D con 4 params (C_x, C_y, R, A). 94.30% MNIST con 3,850 params.
- **Invarianza a resolución:** La parametrización geométrica continua escala de 28×28 a cualquier resolución sin añadir parámetros.

### 4.5 Arquitectura Espectral V11 (tiny-thinker)

LLM de 9.44M params con weight-sharing y kernel Walsh, entrenado en CPU:

| Config | d | k | l | Params | Val Loss | Tiempo/iter |
|---|---|---|---|---|---|---|
| Baseline | 512 | 128 | 6 | 4.36M | 4.5435 | 12.7s |
| **Run 2 (best)** | **1024** | **512** | **8** | **9.44M** | **4.1287** | **37.9s** |

**Hallazgos:** El rango Walsh k importa más que la dimensión d; weight-sharing actúa como regularizador; weight decay = 0 es óptimo.

---

## 5. Ecosistema de Algoritmos Derivados

| Algoritmo | Área | Idea central | Resultado clave |
|---|---|---|---|
| **SeismicDescent** | Optimización global | Deforma el paisaje con ruido correlacionado (RFF) | Rastrigin 5D: 100% éxito; MNIST: 97.90% |
| **SMO** | Optimización/compresión | Comprime momentos de Adam mediante pooling adaptativo | CIFAR-10: +1.44% sobre Adam; MiniGPT: 93% menos memoria |
| **DGE** | Optimización zeroth-order | Perturbaciones por bloques + filtrado Dual Sign-EMA | MNIST 94.16% sin backprop; funciona en INT8/INT4 |
| **PAC Classifier** | Clustering supervisado | Arquetipos que se bifurcan donde hay errores | MNIST: 60,000 → 1,470 arquetipos (99% compresión vs KNN) |
| **CAMEO-ZO** | Optimización zeroth-order | Subespacio de ediciones rango-1 | 3.13× más rápido que MeZO/SPSA en MNIST |
| **RAMA-LoRA** | PEFT | Modulación multiplicativa + actualización aditiva | Error 21.08 vs LoRA 29.64 en benchmark sintético |
| **SOMA** | Orquestador de agentes | Dashboard de ocupación de contexto + memoria L1-L4 | Tareas >2000 turnos sin perder objetivo |

---

## 6. Principios Descubiertos

### 6.1 La fase como principio común
La fase aparece en resonancia, filtros temporales, binding asociativo e invariancia geométrica. La hipótesis unificadora: la fase codifica relaciones relativas — posición, diferencia temporal, compatibilidad entre claves o sintonía con una frecuencia.

### 6.2 La arquitectura debe respetar la geometría del dominio
V55, V97, V101–V104, V117 y V258–V259 apuntan a la misma regla: la estructura del sustrato debe ser compatible con la geometría de los datos. Una base eficiente pero incompatible con el dominio puede ser peor que una base más simple.

### 6.3 La invariancia perfecta puede destruir información
V118 consigue invariancia rotacional casi perfecta descartando fase, pero pierde orientación (~62%). V119 recupera accuracy añadiendo caminos separados. La solución: mantener rutas separadas y dejar que un gate decida cuánto confiar en cada una.

### 6.4 Una reparametrización ortogonal no es una nueva capacidad
V138–V143: si solo se aplica una transformación ortogonal y después se usa distancia L2, el ranking es equivalente al del dominio original por Parseval. La capacidad aparece cuando la arquitectura trata de forma distinta las coordenadas transformadas (truncación, filtrado, cuantización).

### 6.5 Neurogénesis por error
V167–V170: la red añade capas residuales de especialistas solo para corregir errores que sobreviven a capas anteriores. Entrenar → localizar errores → añadir herramienta especializada → congelar → repetir.

### 6.6 Separar descubrimiento y refinamiento
V261–V273 (PID) y stage gating: entrenarlo todo simultáneamente puede destruir la estructura que el sistema acaba de descubrir. Cambiar de fase — alta inercia durante exploración, alta amortiguación durante refinamiento — produce saltos significativos.

---

## 7. Arquitectura Modular Emergente

```
entrada
  ↓
análisis de geometría, familiaridad y dificultad
  ↓
selector de representación
  ├── espacial / convolucional
  ├── espectral / DCT / Walsh
  ├── fase / resonancia
  ├── logarítmica / analítica
  └── simbólica / discreta
  ↓
selector de operador
  ├── mezcla local o convolución
  ├── operador resonante
  ├── DeltaPhase para recall
  ├── experto polimórfico
  └── experto simbólico
  ↓
control de esfuerzo
  ├── ruta rápida
  ├── más memoria
  ├── reflexión espectral
  ├── segundo experto
  └── abstención
  ↓
salida + familiaridad + estimación de error
```

---

## 8. Metodología de Investigación

**Flujo de trabajo:** El autor concibe hipótesis y diseña experimentos; agentes de IA (Claude, GPT, Gemini) proponen la formulación matemática e implementan en PyTorch; el autor ejecuta, diagnostica y documenta.

**Fortalezas metodológicas:**
- 245 experimentos documentados con findings individuales
- Clasificación explícita [ANCLA]/[SEÑAL]/[RUIDO-SOSPECHA]
- Resultados negativos publicados (~30 "anclas negativas")
- Checklist de descarte obligatorio antes de declarar resultado negativo
- Ablations aislados (una variable modificada por vez)
- Código preservado (cada iteración crea archivo nuevo)

**Debilidades:**
- Pocas semillas (mayoría con 1 semilla)
- Escalas toy (MNIST, vocabularios de 32 tokens, secuencias 64–512)
- Sin error estándar formal
- Sin comparación directa con SOTA en igualdad de condiciones
- Terminología a veces excesivamente metafórica

---

## 9. Preguntas Prioritarias (Frontera Abierta)

| Pregunta | Experimento propuesto | Prioridad |
|---|---|---|
| ¿Qué aporta realmente la fase compleja? | Comparar DeltaPhase complejo vs Delta real normalizado a igual estado | Alta |
| ¿Puede el router elegir cadenas completas? | Comparar Lineal→MLP, Log→Harmonic, DCT→Spectral, Phase→DeltaMemory | Alta |
| ¿La confianza debe ser geométrica? | Combinar Atlas de familiaridad + error de reconstrucción + consistencia entre expertos | Media |
| ¿Memoria positiva, contrastiva o ambas? | Comparar acumulativa, DeltaPhase, novedad y dual con decay aprendido | Media |
| ¿La especialización sobrevive a tareas reales? | Probar perturbaciones, dominios cambiantes, lenguaje natural | Alta |
| ¿Qué es barato realmente? | Medir por separado: params entrenables, congelados, estado, FLOPs, tiempo, energía | Alta |

---

## 10. Conclusión

La contribución potencial del proyecto no es una sustitución universal de las redes densas. Es una visión de **arquitectura heterogénea**: una inteligencia eficiente debería reconocer qué clase de problema tiene delante, escoger una representación compatible, aplicar el operador adecuado, recordar solo lo necesario y gastar más cómputo únicamente cuando la situación lo exige.

**DeltaPhase** es actualmente la señal más fuerte del portafolio para el componente de memoria asociativa. Pero el sistema completo probablemente necesite también bases espectrales, expertos analíticos, resonancia, memoria de familiaridad, abstención y control adaptativo del esfuerzo.

El siguiente salto no sería encontrar un martillo más potente, sino construir un buen taller.