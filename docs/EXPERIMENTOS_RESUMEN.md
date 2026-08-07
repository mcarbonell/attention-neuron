# Resumen de Experimentos — Attention-Neuron

>
> Este documento resume todos los experimentos del repositorio. Para cada uno se indica:
> **qué se probó**, **cómo se probó** (setup clave) y **qué se encontró** (hallazgos/resultados).

---

## Índice de Eras

| Era | Rango | # Experimentos | Estado |
|-----|-------|----------------|--------|
| Fundacional | v1–v15 | 13 | ✅ Procesado |
| Revolución de Sustratos | v16–v33 | 16 | ✅ Procesado |
| Matchstick/Haar | v35–v59 | 13 | ✅ Procesado |
| DCT/JPEG | v60–v67 | 6 | ✅ Procesado |
| Espectral & PAC | v69–v89 | 16 | ✅ Procesado |
| Geométrico/Foveación | v90–v99 | 11 | ✅ Procesado |
| Cone/Haar | v101–v109 | 10 | ✅ Procesado |
| Espectral/Holográfico | v110–v146 | 32 | ✅ Procesado |
| Memoria Holográfica | v150–v170 | 14 | ✅ Procesado |
| Resonancia | v171–v199 | 17 | ✅ Procesado |
| Fase/Periódica | v200–v227 | 21 | ✅ Procesado |
| Descubrimiento Matemático | v229–v257 | 30 | ✅ Procesado |
| PID/Control | v258–v274 | 12 | ✅ Procesado |
| Complejo/Fase | v275–v299 | 25 | ⚠️ Parcial (10 inválidos) |
| v300+ | Roadmap | 1 | ✅ Procesado |
> **Nota:** V258–v274 ya están procesados en este documento (sección PID/Control).

---

## Tabla Maestra de Experimentos

| # | Versión | Experimento | Qué se probó | Hallazgo principal |
|---|---------|-------------|--------------|-------------------|
| 1 | v1 | Residual Attention Neuron | Modulación residual sobre sustrato aleatorio (`W_eff = W_init + W_init*M + A + sin(bias)`) | **91.53%** MNIST. La formulación residual estabiliza la optimización. |
| 2 | v2 | Log-Gated Attention Neuron | Gating multiplicativo estrictamente positivo vía `exp(S)` | 87.07%. Forzar positividad no aporta mejora y añade overhead. |
| 3 | v2b | Bounded Gated Attention Neuron | Gating acotado con `tanh` en rango (0,2) | 83.48%. Acotar la modulación penaliza el rendimiento. |
| 4 | v3 | Sparse Attention Neuron | Regularización L1 sobre los pesos de modulación | 85.87%. La red necesita pocas "frecuencias" activas. |
| 5 | v4/v5 | Ablación Pura Multiplicativa vs Aditiva | Aislar la fuente de capacidad: `W_init*M` vs `W_init+A` | **86.64%** vs 42.6%. El gating multiplicativo es el motor principal. |
| 6 | v6 | Dual-Speed Attention Neuron | LRs separados para M (0.001) y A (0.0001) | 86.75%. Frenar A no ayuda; Adam ya equilibra ambos flujos. |
| 7 | v10 | Sinergia DGE + Rank-4 + Incremental | Combinar DGE, rank-4 y batch incremental con patience | **88.79%** con solo 15,482 params. Hito de sinergia. |
| 8 | v12 | Hybrid Attention CNN (CIFAR-10) | CNN híbrida con modulación de canal rank=2 | 26.82% CIFAR-10 (9,785 params). Cuello de botella de capacidad. |
| 9 | v12 | Hybrid Attention CNN (MNIST) | CNN híbrida con modulación de canal y espacial | 83.42% MNIST con solo 6,648 params. Validación del paradigma CNN. |
| 10 | v12b | Hybrid Attention CNN Rank=16 (CIFAR-10) | Aumentar el rank de modulación de canal a 16 | 40.06% CIFAR-10 (76,453 params). El rank es la "perilla" de capacidad. |
| 11 | v13 | Polymorphic Attention Neuron | Dial `alpha` entre agregación SUM y MAX por neurona | 86.51%. Auto-organización de roles; caro (38s/época). |
| 12 | v14 | L2 Polymorphic Attention Neuron | Aproximar MAX con norma L2 vectorizable | 86.46% y ~13s/época. Éxito: rendimiento sin coste. |
| 13 | v15 | Learnable-Lp Attention Neuron | Aprender el exponente `p` de la norma Lp | ~11%. Colapso de gradiente. Vía descartada. |
| 14 | v16 | Over-Parametrized Attention Neuron | Escalar rank=32 y profundidad para buscar 99% MNIST | 98.45%. El rank y la profundidad son determinantes. |
| 15 | v17 | The Colossus | Rank=64, BatchNorm, Data Augmentation | 98.99%. Casi el 99%; la primera capa es el cuello de botella. |
| 16 | v18 | THE ULTIMATUM | Rank-128 en capa 1, Label Smoothing, 60 épocas | **99.09%** MNIST. HITO: estado del arte con sustrato aleatorio congelado. |
| 17 | v19 | The Navigator (CIFAR-10) | CNN 6 capas con modulación de canal rank-32 | **76.76%** CIFAR-10 con 118K params. Nuevo récord. |
| 18 | v22 | The Rosetta Stone | MLP con 4 sustratos aleatorios + dial softmax | 56.72%. El MLP puro no compensa la falta de invariancia espacial. |
| 19 | v23 | The Hybrid | Sensor Rosetta congelado + cerebro plástico | 62.51%. El cuello de botella es la extracción de rasgos. |
| 20 | v24 | The Kaleidoscope | CNN mezclando 4 sustratos por canal (rank=16) | 75.18% con solo 64K params. Récord de eficiencia. |
| 21 | v25 | The Great Arborist | ResNet-18 con árbol binario de 8 sustratos + Mixup | 79.65% antes de fallo del sistema (error 1224). |
| 22 | v25_FAST | The Quick Prism | 3 capas conv, 4 sustratos, LR fijo | 67.74%. Récord de velocidad (48.71% en época 1). |
| 23 | v26 | Perlin Spectrum | Sustratos con ruido Perlin vs ruido blanco | 75.56% (+0.38% vs V24). Sintonía frecuencial autónoma. |
| 24 | v26 | The Prism-ResNet | ResNet-18 + 4 sustratos + rank-16 | **85.94%** CIFAR-10 con 439K params. RÉCORD ABSOLUTO. |
| 25 | v29 | The Splatter-ResNet | Visión continua con Gaussian Splatting 2D | 62.75%. Demuestra visión sin convoluciones discretas. |
| 26 | v30 | The Framer | Soft Window Attention O(H+W) | 58.84% con 50K params. Barato pero plano. |
| 27 | v31 | The Spectrum Library | Biblioteca de 4 espectros de ruido (blanco/Perlin/azul) | 72.01% a mitad de entrenamiento (SGD Nesterov). |
| 28 | v32 | The Broadcaster | Modulación Fan-out (post-activación) | 71.53%. El fan-out no puede esculpir el kernel. |
| 29 | v35 | The Walsh Filter (FWHT Attention) | Filtrado global en dominio de Walsh (FWHT) en vez de convoluciones | **74.04%** CIFAR-10. FWHT es extractor SOTA. |
| 30 | v36 | The Walsh-MNIST MLP | Arquitectura "Zero-Weight" filtrando frecuencias Walsh | **98.54%** MNIST. Walsh es el lenguaje de los trazos. |
| 31 | v37 | Seismic Walsh Optimizer | Optimizador que "tiembla" con patrones de Walsh | 97.25% (V38). El cooling es vital. |
| 32 | v39 | The Banded Walsh Equalizer | Ecualizador de 4 bandas de frecuencia (512 params núcleo) | 93.98% (96% con LR bajo). Compresión extrema. |
| 33 | v40 | The Nano-Walsh Net | Red "Zero-Weight" con 938 params totales | 92.12% (93.50% V40b). Récord de eficiencia. |
| 34 | v41–v48 | Proyecciones Congeladas | Exploración de capa de entrada congelada (parches, contraste) | **98.58%** (V46). Contraste local es clave. |
| 35 | v50 | Stroke Neurons | Curvas de Bézier entrenables como filtros | 97.88% con 35,722 params. 100% caja blanca. |
| 36 | v51 | Matchstick Neurons | Segmentos de línea recta (6 params/neurona) | **98.30%** MNIST. La línea es el "átomo" de la forma. |
| 37 | v52 | Double Matchstick Neurons | Dos segmentos de línea por neurona | 97.52%. Ley de rendimientos decrecientes. |
| 38 | v54 | RGB Matchstick Neurons (CIFAR-10) | Matchsticks con sensibilidad al color RGB | 61.18% CIFAR-10. Señal fuerte en imágenes naturales. |
| 39 | v55 | Symmetry Break & Geometric Blindness | Inicializar todas las neuronas en el centro | 95.71% en 100 épocas. La cobertura inicial es obligatoria. |
| 40 | v57 | Grid Initialization | Inicializar neuronas en rejilla 16x16 | 97.16%. Arranque explosivo (95.84% época 1). |
| 41 | v59 | DCT Attention Neurons | Modulación DCT (JPEG) en vez de pesos densos | 98.12% MNIST con 12.25x compresión. |
| 42 | v60 | Extreme Compression (MNIST) | Solo 4x4 coeficientes DCT por neurona | 93.17% con 16 params/neurona (49x compresión). |
| 43 | v61 | Global DCT Attention (CIFAR-10) | Modulación de frecuencia global en imágenes color | 62.64% CIFAR-10. Campo receptivo global desde capa 1. |
| 44 | v62 | Convolutional DCT Kernels (CIFAR-10) | Sintetizar kernels 8x8 desde coeficientes DCT 4x4 | 72.72% CIFAR-10. Convergencia instantánea (62.79% época 1). |
| 45 | v63 | The All-DCT MLP | Comprimir toda la topología interna del MLP en dominio DCT | 97.59% MNIST con 56.2x compresión (11,914 params). |
| 46 | v64 | The DCT-Transformer (NLP) | Reemplazar FFNs SwiGLU por DCTLinear en Transformer | Loss 6.08 (32x compresión FFN). Frecuencia semántica en lenguaje. |
| 47 | v65 | The "JPEG" of Language | Truncar frecuencias DCT de embeddings de secuencia | El significado reside en bajas frecuencias; generación coarse-to-fine. |
| 48 | v66 | The Fully-JPEG LLM | Comprimir 100% de la topología del Transformer (Q,K,V,O + FFN) | Loss 6.22. La atención es armónica; fin de la memorización bruta. |
| 49 | v67 | The Hybrid Spectral GPT | DCT para atención + Walsh para FFN | Loss 6.31. Interlingua espectral; FFN solo sumas/restas. |
| 50 | v69 | Spectral Interpretability & Modularization | Modularizar en librería + visualizar neuronas DCT 2D | 92.4% MNIST (2,570 params). IA explicable. |
| 51 | v70 | Hierarchical Composition & Visual Atoms | 2 capas: 20 "átomos visuales" DCT + mezclador | 95.96% MNIST (~3,000 params). Representación por partes. |
| 52 | v71 | The Neural Vectorizer | Optimizar trazos Bézier para "dibujar" una imagen | Convergencia <60 épocas. Generación vectorial. |
| 53 | v74 | Archetype Nearest Centroid | Clasificar por distancia MSE a 10 arquetipos | 82.03% (píxel) / 78.66% (vector). Sin entrenar. |
| 54 | v75/v76 | The Purifying Archetype Classifier | Purificar arquetipos aislando errores | **93.50%** con 280 arquetipos. Interpretabilidad absoluta. |
| 55 | v79 | Active Morphing Classifier | Inferencia activa deformando arquetipos (analysis-by-synthesis) | 86.00% sin entrenamiento. Confusión 4↔9. |
| 56 | v81 | PAC + K-NN Voting | Votación Top-K sobre 491 arquetipos | 94.43% (1-NN). K>1 degrada; 1-NN es lo correcto. |
| 57 | v82 | DCT Image Reconstruction | Reconstruir imagen optimizando 64 coeficientes DCT | MSE 0.0347. DCT como "receta" generativa. |
| 58 | v84 | Spectral Basis Comparison | Comparar DCT vs Walsh-Hadamard para reconstrucción | DCT gana en fidelidad; Walsh en hardware. |
| 59 | v85/v86 | The Neural-PAC Prototypes | PAC integrado en red neuronal con DCT neurons | Neurogénesis por reconstrucción; taxonomía de estilos. |
| 60 | v87 | The 16K Mega-Layer Breakthrough | Capa 16,384x16,384 con síntesis espectral FWHT | **65,540x compresión**, 40.2x más rápido. Muerte del memory wall. |
| 61 | v87b | Mega-Layer Learning Validation | Validar que la mega-capa espectral aprende mejor | 4.18x menor MSE que baselines iso-paramétricos. |
| 62 | v87c | Cross-Basis Verification | Verificar FWHT en bases no nativas (Walsh/DCT/aleatoria) | FWHT es prior estructural, no universal. PCA gana. |
| 63 | v87d | Smooth Walsh vs DCT vs Blocky | Comparar bases en señales continuas vs discretas | DCT óptimo para continuas; Walsh para discretas. |
| 64 | v88 | El Hipocampo Holográfico | Memoria O(1) con FWHT + interferencia holográfica | **ÉXITO**: recall 0.4861 tras 51,200 tokens. |
| 65 | v89 | El Cerebelo Espectral | Inferencia dinámica early-exit (Sistema 1/2) | 93.7% de trabajo en vía rápida; 2.2x más rápido. |
| 66 | v90c/d/e | Placa Analógica (Evolución) | Gating dinámico, máscara espectral, resonador holográfico | **97.92%** (V90e). Resonancia > suma lineal. |
| 67 | v90b | Placa Analógica Adaptativa | Mezcla aprendible de agregadores (SUM, VAR, L2, LSE, Walsh) | Pendiente de ejecución. Diversidad matemática. |
| 68 | v93 | Spiral Pixel Ordering | Ordenación espiral centro-fuera + Walsh | 22.97% (3 épocas). +0.85% vs raster. |
| 69 | v93b | Fractal Hierarchical MLP | Entrada multiresolución (promedios jerárquicos) | 97.90% (época 3). Acelera convergencia inicial. |
| 70 | v95 | Log-Polar Spiral Sampling | Muestreo continuo log-polar (foveación analógica) | **98.29%** (época 5). +0.33% vs raster. |
| 71 | v97 | Fourier-Mellin Invariance | Preprocesado RST-invariante (torture test) | 35.20% vs 20.34% raster. +14.86% robustez. |
| 72 | v98 | Invariant Spectral Attention (ISA) | FM + Atención espectral Walsh | 41.43% (pico). Filtrado frecuencial superior. |
| 73 | v98b | DCT vs Walsh en ISA | Comparar DCT vs Walsh sobre firma FM | 41.93% (pico DCT). Walsh más robusto al final. |
| 74 | v99 | Triangular Attention Neuron | Neuronas 1D con solo centro y ancho (2 params) | 79.59% (raster). ~80% con 11k params. |
| 75 | v99b/c | Multi/Omni-View Triangular | Múltiples vistas (3 y 5) de la imagen | **84.21%** (5 vistas). Diagonales informativas. |
| 76 | v101 | Cone Attention Neurons | Conos 2D con 4 params (centro, radio, amplitud) + inhibición | **94.30%** MNIST con 3,850 params. Eficiencia extrema. |
| 77 | v103 | Cone Neurons for Language Modeling | Conos temporales en Transformer (ConeAttn/ConeFFN/FullCone) | +4% loss con 24% menos params. Radios crecen con profundidad. |
| 78 | v103–v106 | The Haar Wavelet Era | Wavelets de Haar para bordes localizados | **96.20%** (V106) con 12.6k params. BatchNorm espectral. |
| 79 | v104 | ConeFFN Radius Collapse | Conos en FFN colapsan a radio ~1 | El FFN es un selector sparse; d=64 es pequeño. |
| 80 | v105 | Is FFN Just a Dimension Gate? | Comparar FFN denso vs NarrowFFN vs DimGate | **NarrowFFN**: +1% con 11.5x menos params. FFN sobreparametrizado. |
| 81 | v106 | ConeAttn + NarrowFFN Combined | Combinar ambas victorias | +10.6% (superaditivo). PEI mayor que baseline. |
| 82 | v107 | Feature Fusion (MNIST) | Fusión de características morfológicas (Islands) + intensidad | 94.70% con 113D. Alta densidad de información. |
| 83 | v107 | Iso-Budget Comparisons | Comparar arquitecturas con presupuesto fijo de params | DimGate no escala con profundidad (colapsable). |
| 84 | v108 | nGPT + ConeAttn | nGPT (hiperesfera) + conos | nGPT converge más lento; ConeAttn mejora nGPT +2.3%. |
| 85 | v109 | Cross-Neuron Comparison | 16 configs: 4 neuronas x 4 representaciones | Triangular+Islands: 80.02% con solo 426 params. |
| 86 | v110 | Tri-Walsh Hybrid | Cerebro-cerebelo: Triangular(Islas) + Walsh(Píxeles) | 93.03% con 1,290 params (~20x compresión). |
| 87 | v111 | Scaled Tri-Walsh Hybrid | Escalar a H=96 | 94.20% con 3,850 params. Rendimientos decrecientes. |
| 88 | v112 | Spiral-Hybrid | Reemplazar raster por espiral log-polar en Walsh | 91.73% (peor). El raster es más "Walsh-friendly". |
| 89 | v113 | Full Morph-Spectral Hybrid | Fusionar Islas+Intensidad+Píxeles | 93.01%. "Menos es más" en ultra-compactos. |
| 90 | v117 | Infinite Resolution Paradox | Aumentar muestras a 32,768 sin subir k | Degrada. Resolución infinita requiere capacidad espectral infinita. |
| 91 | v118 | Spectral Rings | Anillos concéntricos + magnitud FFT | **Invarianza rotacional perfecta** (62.59% en 0/90/180°). |
| 92 | v119 | Invariant Hybrid King | 3 lóbulos: Rings + Islas + Mini-Raster | **92.01%** a 0° con 3,322 params. Robusto a 15°. |
| 93 | v120 | Radical Cosine | Activaciones periódicas (cos/sin) en MLP | Periódicas arrancan más rápido; ReLU gana al final. |
| 94 | v121 | Projection Sinusoids | Proyecciones 1D + moduladores sinusoidales | 88.87% con 5,386 params. Sin pesos espaciales. |
| 95 | v122 | Smooth Walsh Neurons | Walsh de baja resolución + interpolación bilineal | **98.13%** (K=16) supera al Dense con 4x menos params. |
| 96 | v123 | Fair Smooth Comparison | Walsh vs DCT con params igualados | Walsh ama smoothing; DCT ama pureza espectral. |
| 97 | v124 | Micro Walsh K=2/K=4 | Límites de compresión ultra-baja | K=4 smooth: 90.18%. K=2: demasiado borroso. |
| 98 | v125 | Smooth Spectral Adam (SWO) | Comprimir estados de Adam espectralmente | **93.6% RAM saving** con solo -0.77% accuracy. |
| 99 | v126 | Total Spectral Entropy | Arquitectura espectral + optimizador espectral | 51x menos RAM de optimizador (82 KB). 90.40%. |
| 100 | v127 | ARSO | Resolución adaptativa recursiva del optimizador | Paridad con V126. Transición delicada (requiere LR decay). |
| 101 | v128 | LLM Spectral Scanning | Análisis espectral de GPT-2 | **Walsh > FFT/DCT** en compactación de pesos. |
| 102 | v129 | LLM Spectral Pruning | Podar coeficientes Walsh de GPT-2 | **Umbral mágico 50%**: coherente a 2x, colapso a 4x. |
| 103 | v130 | Block-Based Spectral Pruning | Podar por bloques locales | Límite zero-shot estable: 2x. Bloques mejoran semántica. |
| 104 | v131 | Spectral Quantization | Cuantizar 1-bit/2-bit Walsh en GPT-2 | **Colapso total**. La magnitud relativa es crítica. |
| 105 | v132 | Universal Approximation | Poly-Neuron (65p) vs MLP (4.3kp) | Poly generaliza mejor en x²; falla en interacción. |
| 106 | v133 | Interaction Polymorphic | Canal PROD explícito | **1000x mejor** en prod (0.000049). Iguala MLP 4.3k. |
| 107 | v134 | Spectral Cerebellum Polymorph | Banco Walsh en neurona polimórfica | **41x mejor** en 1/x. 289-361 params. |
| 108 | v135 | Cognitive Hierarchy | Fast/Slow thinking con Surprise Gate | Gate 0.4% para prod (fácil); 72% para tan (difícil). |
| 109 | v136 | Escalabilidad y Saturación | Benchmark dim 8192 GPU | 2.6x más rápido. El muro de Adam domina. |
| 110 | v137 | Humillando al MLP | Dimensiones 16k-131k en GPU integrada | **131,072 dims** en 96 MB. MLP colapsa a 32k. |
| 111 | v138 | Memoria Holográfica Espectral | CAM masiva de 131k ítems | **100% precisión con 50% ruido** en 16ms. |
| 112 | v139 | Holographic MNIST | Zero-shot con 60k recuerdos | 92.42%. **Nota: por debajo del baseline 1-NN (96.9%)**. |
| 113 | v140 | Holographic-PAC | Purificar 60k → 203 arquetipos | 92.84% con 295.6x compresión. |
| 114 | v141 | Spectral PAC-V2 | Purificación por pares de confusión | 93.83% con 960 arquetipos. Taxonomía de confusión. |
| 115 | v142 | Refined PAC | Refinar arquetipos con gradientes | **Degrada** (84.39%). La taxonomía > el ajuste. |
| 116 | v143 | CSI Espectral | Auditoría de datos con memoria 131k | **1,423 anomalías** (2.37%). Detecta error real 59915. |
| 117 | v146 | Hybrid Memory Walsh+Islas | Fusión frecuencia + topología | **97.42%** zero-shot (1080D). Visión binocular. |
| 118 | v150 | Resonancia Dual | Guardar 2 versiones de cada dato (120k slots) | **97.68%** zero-shot. Invarianza por redundancia. |
| 119 | v151–v156 | Élite PAC y Cerebro Fluido | Destilar 120k → 30k con EMA update | **97.03%** con 30k slots. Memoria dinámica 2x eficiente. |
| 120 | v157–v161 | Cristales Holográficos | Superposición holográfica (multiplexación, clanes, manifold) | 83.97% con 937x compresión (V161). |
| 121 | v162 | Meta-Abstracción | Clasificar por "ritmo de resonancia" (Meta-Walsh) | 71.32%. Firma estructural única. |
| 122 | v163 | Spectral-FFN | Proyección → Hopfield S^16 → Síntesis | 91.94% con 234x compresión. |
| 123 | v164 | Profundidad y Residuos | 2 capas jerárquicas analizando residuo | 91.99%. El residuo espectral puro es difícil. |
| 124 | v165 | Spectral-MoE | Router global + expertos de clase | **93.27%** con 138x compresión. Especialización clave. |
| 125 | v166 | Auto-Crítica | Análisis por síntesis (reconstruir imagen mental) | 91.58%. Necesita fidelidad altísima. |
| 126 | v163b–v163e | Evolución hacia LLM Espectral | Escalado a 4,096 expertos; MoE extremo 131k | **95.68%** (V163b); 308 tok/s CPU (V163d). |
| 127 | v163f | Stress Test Holográfico | Saturación de memoria sin atención | Colapso rápido. La saliencia es obligatoria. |
| 128 | v163g | Holographic Attention | Peso de saliencia W en token aguja | **100% recall a 4k tokens** con W=20. |
| 129 | v163h | Sentence Recall | Frase de 8 tokens en holograma con ruido | **100% fidelidad**. Roll preserva el orden. |
| 130 | v163j–v163n | Spectral V8.3–V8.6 | Optimización del motor espectral (residencia, C++, GPU) | **10.63 tok/s GPU** (V8.6). 2500x compresión. |
| 131 | v167–v170 | Auto-Arquitecto | Neurogénesis residual + capas identidad | **96.08%** (V170). Red que se auto-cultiva. |
| 132 | v168 | Vectorización Holográfica | Flash-Hologram + MoE jerárquico por clanes | **6.8x speedup**; 1.1T params equivalentes. |
| 133 | v171–v189 | Era de la Resonancia | Osciladores armónicos + votación física | **96.12%** (V186) con 75k; 87.34% (V188) con 1.1k. |
| 134 | v190 | Structural Generalization | Benchmark OOD de funciones matemáticas | Poly (161p) compite con MLP (132k) en extrapolación. |
| 135 | v191 | Log-Polymorphic Interaction | Rama logarítmica para productos/divisiones | **64x más estable** en div (ratio 1,690 vs 109,000). |
| 136 | v192 | Resonant-Log-Polymorphic | Unificar resonancia + log + estructural | **32,000x mejor** en Schwefel (ratio 25.1). |
| 137 | v193 | Deep Polymorphism | Profundidad multicapa polimórfica | Ratio 0.998 en Schwefel (comprensión perfecta). |
| 138 | v194 | Modulus Challenge | Función módulo (discontinuidades) | MLP-Huge gana local; Poly 3x más estable OOD. |
| 139 | v195 | Discontinuity Branch | Rama sawtooth con STE | **400x menos params** que MLP-Huge con precisión similar. |
| 140 | v196 | Recursive Compression Paradox | Comprimir coeficientes de compresión previa | **Fallo masivo** (MSE 303). Decorrelación. |
| 141 | v197 | Lateral Interaction | Neuronas hijas con operaciones simbólicas | Loss 1.65 en (x*y)%(x+y) con 967 params. |
| 142 | v198 | Entropy-Spectral Hybrid | Huffman sin pérdida sobre coeficientes | **21.28x compresión total**; 85x a escala GB. |
| 143 | v199 | Competence-Based MoE | Gater que predice el mejor experto | MoE 0.186 (mejor que expertos individuales). |
| 144 | v200 | Bounded Parameters | Pesos acotados [-1,1] + factor de escala | Loss 1.42e-05. Listo para 8-bit. |
| 145 | v202 | Resonancia de Fase (XOR) | Interferencia de fase + ReLU + BCE | **XOR resuelto con 41 params** (acc 100%). |
| 146 | v203 | Resonancia a MNIST | FastResonantLayer (identidad trigonométrica) | **96.22%** en 5 épocas (203k params). |
| 147 | v204/v205 | Firewall Biológico | Phase Jitter y vacunación por ruido | **87.72%** con ruido std=1.0 (escala π/4). |
| 148 | v206 | Resonancia Espectral | DCT fijo + resonancia de fase | **91.96%** con 19.2k params (10x compresión). |
| 149 | v207 | Módulo con Resonancia | Módulo con cosenos | 0.0612 local; OOD 29.62. Frecuencia depende de y. |
| 150 | v208 | Explosión Multiplicativa | Log-exp + cosenos para módulo | **Colapso OOD** (551). Fenómeno de Gibbs. |
| 151 | v209 | Sawtooth Resonance | Oscilador discontinuo nativo | Mejor ratio (28.4) pero no ajusta local (1.07). |
| 152 | v210 | Neurona Analítica Discreta | Pesos enteros con STE | **Mejor OOD** (13.15) con 7.5k params. Ratio 19.5. |
| 153 | v212 | Optimización Simbólica DGE | DGE a través de operadores no-diferenciables | **Ratio 8.61** (récord de estabilidad) con 177 params. |
| 154 | v213 | Activation Bridge | Gradientes sintéticos a bloque simbólico | **Fallo** (Sísifo). Adam no puede con paisajes periódicos. |
| 155 | v214 | El Colapso de los Expertos | Soft MoE lineal vs logarítmico | **Collusion**: expertos se anulan mutuamente. |
| 156 | v214–v217 | Batalla de Espacios | MoE Honesto (competición darwiniana) | Multiplicación→Log; Resta→Lineal. Mapa de verdad. |
| 157 | v218 | Compositional CAN | 2 capas con MoE honesto (sin(x·y)) | **OOD 0.87** (333% mejor). Log→Har descubierto. |
| 158 | v219 | Conscious CAN | Cabeza de confianza (auto-confianza) | **Efecto arrogancia**: confianza colapsa en OOD. |
| 159 | v220 | Familiarity Atlas | Memoria de prototipos espectrales | **7.5x discriminación** normal vs rotado. |
| 160 | v221 | Safe Classifier | Abstención por distancia al Atlas | **100% precisión filtrada** (33% abstinencia). |
| 161 | v222 | Spectral Diffusion | Difusión en dominio DCT | Coherente pero 10 épocas insuficientes. |
| 162 | v224 | Neurona Periódica vs ReLU | σ(tan(x)) con 4 params vs MLP 2,241 | **PEI 4.9x superior** (1.3275 vs 0.2708). |
| 163 | v225 | Straight Periodic | Corrección polinómica de la rampa | **0.019 MSE con 8 params** (vs 0.014 MLP). |
| 164 | v227 | MNIST Espectral-Periódica | DCT-2D + StraightPeriodic | **85.83% con 1,034 params**. |
| 165 | v229 | Cuantización Espectral | Walsh jerárquico 8/4-bit vs RTN espacial | **-4% MSE, +62% outliers** en GPT-2. |
| 166 | v235 | Curva de Elasticidad Espectral | Barrido Top-K pruning en GPT-2 | **40% ahorro** con +5 PPL (punto óptimo). |
| 167 | v236 | Robustez Compresión Espectral | Poda Top-K en dataset diverso | **30% ahorro estable** (+1.31 PPL). |
| 168 | v239 | Firmas Espectrales | Exponer k=8 componentes individuales | **Destructivo** (71.49% vs 78.50% suma). |
| 169 | v240 | Diferenciabilidad Mixta | MoE con Adam (analítico) + DGE (simbólico) | **Train MSE 0.0598**; PEI 4.57. |
| 170 | v242 | Adam-DS | Consistencia de signo temporal (DS-EMA) | **98.94%** (+0.14%); loss -10.6%. |
| 171 | v243 | Lion-DS | Signo del momentum, sin varianza | **99.38%** con 5 bytes/p (37.5% menos RAM). |
| 172 | v244 | Muon Superiority | Ortogonalización de actualización | **99.60%** (récord). Ortogonalidad es clave. |
| 173 | v244 | Sign-DS | Solo signo del gradiente + DS | 98.18% con 2 bytes/p (75% menos RAM). |
| 174 | v244 | Duelo Final de Optimizadores | Adam/Lion/RMSprop/SGD/Sign-DS | **SGD+Momentum 99.69%** (mejor balance). |
| 175 | v245 | Noise Robustness | Espectral vs MLP con ruido de labels | Espectral PEI >21 (6x menos params). |
| 176 | v246 | Augmented Features | Neurona lineal + expansión de base | **x³ descubierto** (6.34 vs 158 MLP). |
| 177 | v248 | Pruned Approximator | Aumento + L1 + poda agresiva | **Descubrimiento perfecto** (10^-12 a 10^-19). |
| 178 | v249 | Deep Scientific Network | Composición de leyes (g(f(x))) | **Gaussiana/sin-square reconstruidas**. |
| 179 | v250 | Spectral Hysteresis | Memoria EMA de representaciones | **-7.6% en clusters** (filtro destructivo). |
| 180 | v251 | Multiplicative Gating | Gating por neurona con pesos congelados | **77.02% con 522 params** (780x reducción). |
| 181 | v251b | Gating Sweep | Escalado de D en gating congelado | **89.30% con 4,106 params** (D=4096). |
| 182 | v251c | Deep Gating | 2 capas de proyecciones congeladas | **91.68%** (D=4096). Profundidad ayuda. |
| 183 | v251d | LR Sweep | Sensibilidad al LR | LR 5e-3 a 1e-2 óptimo. |
| 184 | v251e | Scheduled Gating | OneCycleLR con LR alto | **93.89% con 4,106 params**. |
| 185 | v251f | Gate Sparsity | Weight decay en gates | **88.49%** (-5.4%). La inteligencia es colectiva. |
| 186 | v251g | Round-Robin | Actualizar 1 capa por batch | [TBD]. -66% coste por batch. |
| 187 | v251j | LR Switching | Warmup gates + refinamiento pesos | **97.63%** (fase 2). Salto +7% al cambiar. |
| 188 | v251k | WD Impact | Weight decay en gating | **NUNCA usar WD** (-5.29%). |
| 189 | v251l | Oligarchy Hypothesis | Inicialización 0 + SiLU | **94.27%**; ~1965 gates efectivos. |
| 190 | v253 | Ternary Weights + Float Gating | Pesos ternarios congelados + gates float | **94.74%** con 4,106 params. |
| 191 | v254 | Binary Inhibition | {0,1} vs {-1,0,1} | **Colapso binario** (41.4%). Inhibición esencial. |
| 192 | v255 | Full Ternary | Pesos + gates ternarios (sin multiplicaciones) | **82.20%**; 56.4% sparsity. |
| 193 | v256 | Ternary CNN | Kernels 5x5 ternarios + gates canal | **85.07% con 394 params**. |
| 194 | v257 | Ternary CNN + GAP | GAP en vez de FC aplanado | **83.40%**; PEI 28.6. |
| 195 | v258 | Spectrum-Gated Transformer (SGT) | FWHT en vez de atención + pesos ternarios congelados | **71.20%** con 1,546 params. |
| 196 | v259 | Residual SGT (RSGT) | + Residual connections, dim 1024 | 72.12%. Los residuales son necesarios. |
| 197 | v260 | High-Res PSGT | Parches 2x2 (256 tokens) + posicionales + 4 bloques | **91.69%** con 1,290 params (PEI 29.4). |
| 198 | v261 | PID Optimizer (Industrial Miracle) | Controlador PID (Kp/Ki/Kd) como optimizador | **98.47% MNIST** (Ki=150). 6.5x loss menor. |
| 199 | v265 | Universal PID Benchmark | PID en MLP estándar + BatchNorm | **98.41%** (Ki=100, Clip 10). |
| 200 | v267 | PID Hyperparameter Sweep | Grid 27 configs (Kp, Ki, Kd) | **98.27%** (Ki=100 dominante). |
| 201 | v268 | CIFAR-10 PID Instability | PID (1,100,10) en CIFAR-10 | **Falla** (-0.59%). Ki alto = inestable con ruido. |
| 202 | v269 | Extreme Integral Gain | Ki=500 en CIFAR-10 | **75.54%** (+0.43% vs Adam). Efecto "cargo train". |
| 203 | v271 | Breaking 80% Barrier | Ki=1000 en CNN anchas (T4) | **80.41%** (época 9). Recuperación elástica. |
| 204 | v272 | Damping Trade-off | Ki=1000, Kd=10 (estabilizador) | 79.27%. Estable pero suprime picos. |
| 205 | v273 | Phase Shift Discovery | Fase 1: Ki=1000; Fase 2: Ki=100,Kd=20 | **83.25%**. Salto +5.29% al cambiar. |
| 206 | v274 | Autonomous Industrial Pilot | Trigger automático de phase shift | **82.71%**. Salto +6.33% al cambiar. |
| 207 | v275 | Complex-Valued MLP (Wave Interference) | CVNN con ModReLU para interferencia | **PEI 1.9718** vs 1.5530 real. Fase+amplitud superior. |
| 208 | v276 | Complex MNIST FFT | CVNN con FFT 2D como entrada | **95.43% con 101K params** (PEI 19.05 vs 18.46 real). |
| 209 | v277 | Complex Transformer (Hermitian Attention) | Q·K^H para secuencias periódicas | **Loss 0.6466** (4x mejor que real). PEI 2.06 vs 1.54. |
| 210 | v278 | Phase Spectral Mixer | Fase analítica e^(iφ) en SpectralMixer | **100% acc en Ep 1** (Single Spike Half). La fase codifica posición. |
| 211 | v279 | Phase LM on Real Text | ComplexFFT vs Walsh en Tiny Shakespeare | **ComplexFFT_noPE 0.0439** vs Walsh_PE 0.1699 (4x mejor). |
| 212 | v280 | Causal Phase LM | Zero-padding para causalidad | 0.0171 (leakage detectado). |
| 213 | v281 | True Causal Phase LM | FFT causal verdadero (h[t>0]=0) | **1.7222** (causal real). Walsh no puede ser causal. |
| 214 | v282 | Ultimate Phase-nGPT | CausalFFT + NarrowFFN + nGPT | **116,870 params** (19.2% baseline). PPL 5.35 vs 4.77. |
| 215 | v283 | Matrix-Free Phase-nGPT | Reemplazar proyecciones lineales por WalshLinear | **42,764 params**, Loss 1.6581 (supera al denso 116K). |
| 216 | v284 | Spherical Loss & Phase Reg | τ aprendible + regularización continuidad fase | **1.7664** con 24K params. τ escala de 10→43.5. |
| 217 | v285 | Fourier Hippocampus | K_mem=16 frecuencias bajas por capa | **99.8% exact match en Ep 3**. Contexto infinito O(1) RAM. |
| 218 | v286 | Poincaré Attention | Atención en disco de Poincaré (Soft-Tanh) | **38.37%** con d=4 (vs 35.35% Euclidiana). |
| 219 | v287 | Conformal Optics | Pesos como proyección de textura compleja | Concepto: holomorfismo como inductor de suavidad. |
| 220 | v288 | Spectral Compression Zero-Shot | DCT "BMP a JPG" en GPT-2 | Poda DCT 10% → 95.41 PPL (vs 89.58). |
| 221 | v289 | Spectral Quantization | Cuantización DCT canal-por-canal | RTN 2-bit: 2710 PPL; DCT 2-bit: mejora marginal. |
| 222 | v290 | Permutation Spectral | PCA 1D / Greedy TSP / Fiedler | Permutación preserva PPL (±1.71e-5). DCT post-orden mejora. |
| 223 | v291 | Oligarchy Validation | Fashion-MNIST, 3 capas, CIFAR-10 | ~51% gates activos (D=4096). Hipótesis robusta. |
| 224 | v292 | MQAR Spectral Mixer | CausalComplexFFT en MQAR | **INVÁLIDO** (harness error pre-v298). |
| 225 | v293 | Holographic Phase Recall | HRR para MQAR | **INVÁLIDO** (harness error pre-v298). |
| 226 | v294 | Multihead Holographic | Holographic multicabeza | **INVÁLIDO** (harness error pre-v298). |
| 227 | v295 | Phase Sharpener | Armónicos 2θ,4θ,8θ | **INVÁLIDO** (harness error pre-v298). |
| 228 | v296 | Causal Norm | Normalización RetNet/RWKV | **INVÁLIDO** (harness error pre-v298). |
| 229 | v297 | Phase Softmax | Softmax selectivo por contenido | **INVÁLIDO** (harness error pre-v298). |
| 230 | v298 | Delta Phase MQAR (ANCLA) | Regla Delta matricial + fase compleja | **99.95% en Ep 2** (O(N) = O(N² Softmax). ARNÉS CORREGIDO. |
| 231 | v299 | Capacity Frontier | Complejo vs Real con iso-floats | **95.98% a 64 pares** (Real: 73.14%). Superioridad demostrada. |
| 232 | v300 | Capacity Scaling (en curso) | Barrido d_k ∈ {32,64,128}, pares 32–256 | **En ejecución**. Mide capacity frontier Complejo vs Real vs Softmax. |
| 233 | v301–v306 | Roadmap Post-V300 | Phase Softmax, Dynamic Decay, Dual Memory, Port tiny-thinker, Quant 4-bit, TSP Walsh | **Planificado**. V304 (port a lenguaje) es crítico. |

---

## Detalle por Era

### Era Fundacional (v1–v15)

> **Tema:** validación del concepto de *Attention Neuron* — modular sustratos aleatorios congelados mediante gating de bajo rango.
> **Hito clave:** la formulación **residual** (V1) se consolida como baseline; el gating multiplicativo es el motor del aprendizaje (ablación V4/V5).

#### V1 — Residual Attention Neuron
- **Qué se probó:** Formular la modulación como corrección residual sobre el sustrato aleatorio (`W_eff = W_init + W_init*M + A + sin(bias)`), inicializando `M` cerca de 0 para partir del "ruido base" puro.
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, `mask_prob=0.5`.
- **Resultado principal:** **91.53%** accuracy (vs 86.64% de V4 Pure Multiplicativo y ~88.80% de V10e).
- **Hallazgo:** La formulación residual produce una curva de aprendizaje muy suave y se consolida como baseline robusto; la red "enciende/apaga" conexiones del sustrato sin shocks de inicialización.

#### V2 — Log-Gated Attention Neuron
- **Qué se probó:** Parametrizar el gating multiplicativo con `exp(S)` (`W_eff = W_init * exp(S) + A + sin(bias)`) para forzar un factor estrictamente positivo.
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, `mask_prob=0.5`.
- **Resultado principal:** 87.07% (~13.0s/época) vs 87.61% de V1 (~11.1s/época).
- **Hallazgo:** Forzar positividad estricta no aporta mejora tangible en 10 épocas y añade overhead computacional; variante descartada.

#### V2b — Gating Acotado (Tanh)
- **Qué se probó:** Restringir el factor de gating al rango (0, 2) usando `W_eff = W_init * (1 + alpha * tanh(S)) + A + sin(bias)` con `alpha = 1.0`.
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, `mask_prob=0.5`.
- **Resultado principal:** 83.48% (vs 87.61% de V1).
- **Hallazgo:** Acotar fuertemente la magnitud de la modulación penaliza el rendimiento; la arquitectura necesita libertad para aplicar factores multiplicativos grandes o negativos.

#### V3 — Sparse Attention Neuron
- **Qué se probó:** Añadir penalización L1 explícita sobre los parámetros de modulación (`L1_LAMBDA * L1_Norm(delta_m)` con `lambda = 1e-4`) para forzar esparcidad.
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, sobre arquitectura base V1.
- **Resultado principal:** 85.87% (vs 87.61% de V1 sin regularización).
- **Hallazgo:** Bajo alta esparcidad el modelo retiene ~86% de capacidad: la red solo necesita activar pocas "frecuencias" del sustrato.

#### V4 vs V5 — Ablación Pura Multiplicativa vs Aditiva
- **Qué se probó:** Aislar la fuente de capacidad de la arquitectura comparando el extremo multiplicativo (`W_eff = W_init * M + sin(bias)`) vs el extremo aditivo (`W_eff = W_init + A + sin(bias)`).
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, `mask_prob=0.5`.
- **Resultado principal:** **V4: 86.64%** vs **V5: 42.6%**.
- **Hallazgo:** "Prueba del algodón" de la tesis: la corrección aditiva de bajo rango tipo LoRA es insuficiente (42.6%), mientras que el gating multiplicativo extrae casi todo el poder representacional (86.64%).

#### V6 — Dual-Speed Attention Neuron
- **Qué se probó:** Entrenar la parte multiplicativa (`M`) con lr=0.001 y la parte aditiva (`A`) con lr=0.0001 (10x menor) para evitar interferencias.
- **Setup:** MNIST, 10 épocas, Adam, `rank=2`, `mask_prob=0.5`.
- **Resultado principal:** 86.75% (vs 87.61% de V1 con LR uniforme).
- **Hallazgo:** Ralentizar la corrección aditiva retrasa el arranque (18% vs 24% en época 1). La modulación topológica y el ajuste fino se benefician de la misma escala temporal.

#### V10 — El Triunfo de la Sinergia (DGE + Rank-4 + Incremental)
- **Qué se probó:** Combinar el optimizador DGE con `rank=4` y strategy incremental de batch (8 → 512) con patience counter.
- **Setup:** MNIST, batch incremental hasta 8192, máscara estocástica 20%, 15,482 params entrenables.
- **Resultado principal:** **88.79%** (vs 79.14% de batch fijo y 87.90% de rank-2 incremental).
- **Hallazgo:** Hito de sinergia: el rank y el batch incremental son multiplicativos en rendimiento; DGE optimiza a través de topologías dinámicas y ruidosas.

#### V12 — Hybrid Attention CNN (MNIST) + Escalado a CIFAR-10
- **Qué se probó:** Trasladar la Attention Neuron a CNNs modulando un kernel convolucional aleatorio congelado (modulación de canal rank=2 + modulación espacial 3x3).
- **Setup (MNIST):** 2 capas conv híbridas (16, 32 canales) + lineal residual; 6,648 params; 10 épocas.
- **Resultado principal:** **83.42%** MNIST. Escalado a CIFAR-10: **26.82%** con 9,785 params (10 épocas, 3 capas conv 32/64/128).
- **Hallazgo:** El paradigma "inteligencia = gating sobre ruido congelado" funciona también en el dominio espacial. En CIFAR-10 hay cuello de botella de capacidad por el bajo rank.

#### V12b — Hybrid Attention CNN Rank=16 (CIFAR-10)
- **Qué se probó:** Subir el rango de la modulación de canal de 2 a 16 para desbloquear capacidad en CIFAR-10.
- **Setup:** CIFAR-10, 3 capas conv híbridas (32/64/128), 76,453 params, 10 épocas, Adam lr=0.001, data augmentation básico.
- **Resultado principal:** **40.06%** (vs 26.82% de rank=2, misma configuración).
- **Hallazgo:** El rank es la "perilla" de capacidad: +13 puntos con 7.8x más parámetros, aún muy por debajo de una ResNet-18 (11M params).

#### V13 — Polymorphic Attention Neuron ("El Dial Neuronal")
- **Qué se probó:** Dar a cada neurona un dial `alpha` entrenable entre agregación SUM y agregación MAX (`y_eff = alpha * y_sum + (1-alpha) * y_max`).
- **Setup:** MNIST, 10 épocas, Adam, arquitectura base V1.
- **Resultado principal:** 86.51% (~38s/época, 3.5x más lento que V1).
- **Hallazgo:** Auto-organización de roles: la capa oculta crea un ecosistema mixto (225 sumadoras, 122 MAX, 165 híbridas) y la capa de salida elige 100% SUM. El gradiente fluye, pero el coste del `max` exacto es prohibitivo.

#### V14 — L2 Polymorphic Attention Neuron
- **Qué se probó:** Reemplazar el `max` exacto por la norma L2 vectorizable (`y_l2 = sqrt(X^2 @ (W^2)^T)`) como proxy suave del detector de rasgos dominantes.
- **Setup:** MNIST, 10 épocas, Adam.
- **Resultado principal:** **86.46%** a ~13s/época (vs 38s de V13, ~11s de V1).
- **Hallazgo:** Éxito rotundo: misma precisión que MAX con coste casi nativo. La capa oculta sigue auto-organizándose (211 sumadoras, 65 L2, 236 híbridas) y la salida elige 100% SUM.

#### V15 — Learnable-Lp Attention Neuron
- **Qué se probó:** Aprender directamente el exponente `p` de una norma Lp generalizada (`p = 1 + softplus(rho)`) para que la red elija su álgebra de agregación.
- **Setup:** MNIST, 10 épocas, Adam.
- **Resultado principal:** ~11% (colapso de gradiente).
- **Hallazgo:** Elevar tensores a potencias dinámicas rompe la retropropagación (inestabilidad numérica extrema) y la restauración de signo con `sign()` no es diferenciable. La vía V14 (dial interpolador entre funciones estables) es la correcta.

### Era Revolución de Sustratos (v16–v33)

> **Tema:** escalado de la *Attention Neuron* hacia el estado del arte. Se pasa de modular un solo sustrato a mezclar **bibliotecas de sustratos aleatorios** (Alquimia de Sustratos) y se conquista MNIST (99%) y CIFAR-10 (85.94%).
> **Hito clave:** V18 logra **99.09%** en MNIST; V26 Prism-ResNet logra **85.94%** en CIFAR-10 con solo ~4% de parámetros entrenables.

#### V16 — Over-Parametrized Attention Neuron
- **Qué se probó:** Alcanzar el 99% en MNIST escalando rank y profundidad (3 capas 784→1024→1024→10, rank=32) con modulación dual y estabilización (LayerNorm + Dropout).
- **Setup:** MNIST, 30 épocas, AdamW + OneCycleLR, 319,134 params entrenables (~17.1% de un MLP equivalente).
- **Resultado principal:** **98.45%** (supera el récord previo de ~94.4% en 4 puntos).
- **Hallazgo:** El rank y la profundidad son los factores determinantes de la capacidad. El bajo training loss (0.0023) indica que la red memoriza; falta generalización para el 99%.

#### V17 — The Colossus
- **Qué se probó:** Romper la barrera del 99% aumentando capacidad (rank=64, capa inicial 2048) y añadiendo BatchNorm1d + Data Augmentation (Rotation + Affine).
- **Setup:** MNIST, 40 épocas, OneCycleLR, 897,310 params.
- **Resultado principal:** **98.99%** (mejor en la última época).
- **Hallazgo:** Casi éxito. La primera capa (784→2048) es donde reside la extracción de rasgos; incrementar su rank podría ser la clave final.

#### V18 — THE ULTIMATUM (MISSION ACCOMPLISHED)
- **Qué se probó:** El asalto final al 99% con Rank-128 en la primera capa, Data Augmentation agresivo, Label Smoothing y OneCycleLR de 60 épocas.
- **Setup:** MNIST, 60 épocas, 1,259,806 params entrenables, ~1,860,000 pesos congelados (sustrato 100% aleatorio).
- **Resultado principal:** **99.09%** (época 60/60).
- **Hallazgo:** HITO. Se demuestra que el aprendizaje reside en la modulación, no en los valores absolutos de los pesos iniciales. El Label Smoothing evitó el overfitting de V16.

#### V19 — The Navigator (CIFAR-10)
- **Qué se probó:** Aplicar el principio de modulación de bajo rango a sustratos convolucionales 3x3 aleatorios en CIFAR-10 (NavigatorNet: 6 capas Conv + 1 Linear, modulación dual por canal rank-32).
- **Setup:** CIFAR-10, 50 épocas, OneCycleLR, 118,238 params entrenables, ~600,000 pesos congelados.
- **Resultado principal:** **76.76%** (nuevo récord absoluto en CIFAR-10 para Attention Neurons).
- **Hallazgo:** Un kernel 3x3 aleatorio bien escalado contiene suficientes rasgos de bajo nivel. Sintonizar "qué canal habla con qué canal" es más importante que el contenido exacto del kernel.

#### V22 — The Rosetta Stone (CIFAR-10 MLP)
- **Qué se probó:** Si un MLP puro puede competir en visión dotándolo de una "biblioteca" de 4 sustratos aleatorios (Fan-in x4) y un dial de atención softmax para mezclarlos.
- **Setup:** CIFAR-10, RosettaStoneNet (3 capas MLP), 612,038 params entrenables, ~8,400,000 congelados.
- **Resultado principal:** **56.72%** (supera el récord previo de 40% en +16 puntos).
- **Hallazgo:** La red usa los 4 sustratos de forma equitativa (~25% cada uno), pero el MLP puro no puede compensar la falta de invariancia espacial de la convolución.

#### V23 — The Hybrid (CIFAR-10)
- **Qué se probó:** Determinar si el cuello de botella del MLP es la extracción de rasgos o la lógica de decisión: capa 1 Rosetta congelada (sensor) + capas 2-3 plásticas (cerebro entrenable).
- **Setup:** CIFAR-10, 2,452,490 params entrenables (~29.2% de un MLP denso), ~6,300,000 congelados, 50 épocas.
- **Resultado principal:** **62.51%** (+5.79% vs V22).
- **Hallazgo:** Un cerebro plástico procesa el sensor Rosetta mucho más eficientemente. El cuello de botella es la extracción de rasgos, no la decisión. La CNN (V19) sigue superior por su sesgo inductivo.

#### V24 — The Kaleidoscope (Efficiency Record)
- **Qué se probó:** CNN donde cada canal mezcla 4 universos de kernels 3x3 aleatorios fijos mediante dial Softmax con modulación rank=16.
- **Setup:** CIFAR-10, KaleidoscopeNet (6 capas Conv), 64,062 params entrenables (mínimo histórico), 50 épocas, OneCycleLR.
- **Resultado principal:** **75.18%** (eficiencia 1.17% por cada mil parámetros).
- **Hallazgo:** Es más eficiente mezclar múltiples sustratos que modular uno solo con más rango. El "sustrato rico" actúa como regularizador natural.

#### V25 — The Great Arborist (Obituario)
- **Qué se probó:** "Sintonía Dendrítica" en ResNet-18: 8 sustratos aleatorios por capa mezclados mediante un árbol binario de 7 diales entrenables por canal, con Mixup augmentation.
- **Setup:** CIFAR-10, ArboristResNet18, 681,226 params (~6.1% de ResNet-18), Mixup α=1.0, 100 épocas planeadas.
- **Resultado principal:** **79.65%** (época 29) antes de fallo del sistema (error 1224 al sobrescribir checkpoint).
- **Hallazgo:** El árbol jerárquico de sustratos es superior a modular uno solo. La arquitectura residual es obligatoria para escalar la alquimia. El fallo fue un bloqueo de archivo del SO, no del modelo.

#### V25_FAST — The Quick Prism (REAL DATA)
- **Qué se probó:** Arquitectura simplificada de 3 capas conv (64/128/256) con mezcla plana de 4 sustratos y LR fijo 0.003, sin scheduler.
- **Setup:** CIFAR-10, FastPrismNet, AdamW sin scheduler, 20 épocas.
- **Resultado principal:** **67.74%** (época 17); récord de velocidad con **48.71%** en época 1.
- **Hallazgo:** La mezcla de sustratos permite "ver" casi de inmediato, pero requiere OneCycleLR para consolidar rasgos finos (oscilación final sin scheduler).

#### V26 — Perlin Spectrum (Ruido Correlacionado vs Blanco)
- **Qué se probó:** Inicializar los sustratos con ruido Perlin estructurado (escalas 0.3/0.6/1.2/2.4) en lugar de ruido blanco puro, para dar un mejor prior espacial.
- **Setup:** CIFAR-10, PerlinSpectrumNet (6 capas conv, kernel 5 en Conv1), 64,062 params, 50 épocas, OneCycleLR.
- **Resultado principal:** **75.56%** (+0.38% vs V24 con ruido blanco, mismos params).
- **Hallazgo:** El ruido Perlin vence al blanco. La red aprende autónomamente la jerarquía del córtex visual: alta frecuencia en capas tempranas (bordes), baja frecuencia en capas profundas (formas globales).

#### V26 — The Prism-ResNet (RÉCORD ABSOLUTO)
- **Qué se probó:** Fusionar ResNet-18 con la Alquimia de Sustratos: 4 universos de ruido blanco congelados por capa + dial Softmax + modulación rank-16.
- **Setup:** CIFAR-10, PrismResNet (18 capas residuales), 439,850 params (~4% de ResNet-18), 50 épocas, OneCycleLR.
- **Resultado principal:** **85.94%** (nuevo SOTA interno, +9.18% vs V19).
- **Hallazgo:** La topología de la atención (dónde y cómo mirar el ruido) es suficiente para igualar redes clásicas masivas. La sinergia residual + alquimia permite que el gradiente fluya limpio hasta la primera capa.

#### V29 — The Splatter-ResNet (Visión Continua)
- **Qué se probó:** Eliminar las convoluciones discretas 3x3 a favor de Gaussian Splatting 2D: óvalos paramétricos continuos (centro, dispersión, rotación, amplitud) + cerebro ResNet 1x1.
- **Setup:** CIFAR-10, 671,146 params, 4 splats por canal (1024 óvalos), 50 épocas, AdamW + OneCycleLR.
- **Resultado principal:** **62.75%** (loss 0.6277).
- **Hallazgo:** Las redes profundas no necesitan grilla discreta para visión compleja. Los óvalos capturan estructura macro pero son "borrosos" para altas frecuencias (texturas finas), limitando el techo.

#### V30 — The Framer (Soft Window Attention)
- **Qué se probó:** Visión continua con "Ventanas Suaves": máscaras sigmoideas separables 1D para cajas delimitadoras diferenciables, reduciendo la extracción a O(H+W).
- **Setup:** CIFAR-10, 50,570 params, 4 ventanas por canal, clasificador MLP (256→128→10), 50 épocas.
- **Resultado principal:** **58.84%** (~25s/época).
- **Hallazgo:** El mecanismo Soft Window funciona y es baratísimo, pero un solo nivel de extracción no compite con una jerarquía profunda (V24: 75.18%). Requiere apilarse en bloques residuales.

#### V31 — The Spectrum Library (White, Perlin, Blue)
- **Qué se probó:** Dar a la red un "menú" de 4 espectros de ruido fijos (blanco, Perlin 0.5, Perlin 1.5, azul) mezclados por Softmax por canal + modulación rank-16, sobre ResNet.
- **Setup:** CIFAR-10, 439,850 params, AdamW (luego SGD Nesterov por fallback DirectML), 50 épocas.
- **Resultado principal:** 46.57% (época 3, AdamW); **72.01%** (época 25, SGD Nesterov).
- **Hallazgo:** La biblioteca de espectros generaliza bien. El parche V31b (SGD Nesterov) resolvió el cuello de botella de DirectML (`aten::lerp` en CPU). Potencial de 80-85% con annealing.

#### V32 — The Broadcaster (Fan-out Modulation)
- **Qué se probó:** Congelar el 100% de los pesos convolucionales (Fan-in) y aplicar la mezcla de sustratos exclusivamente sobre las activaciones de salida (Fan-out) con Gain/Bias.
- **Setup:** CIFAR-10, BroadcasterResNet (18 capas), 210,186 params entrenables, ~44M congelados, GPU DirectML.
- **Resultado principal:** **71.53%** (vs 85.94% de V26).
- **Hallazgo:** Experimento de control perfecto: la modulación Fan-out no puede alterar la geometría del filtro aleatorio (si un kernel no detecta un borde, amplificarlo no lo crea). La alquimia debe ocurrir en el dominio de los pesos (Fan-in).

### Era Matchstick/Haar (v35–v59)

> **Tema:** transición de la "Alquimia de Sustratos" hacia la **IA de Resonancia**: filtrado en dominios ortogonales (Walsh, DCT) y neuronas geométricas (trazos, cerillas). Se abandona la convolución espacial discreta.
> **Hito clave:** V35 demuestra que la FWHT es un extractor SOTA (74% CIFAR-10); V36 logra 98.54% MNIST con arquitectura "Zero-Weight"; V51 (Matchstick) logra 98.30% con solo 6 params/neurona.

#### V35 — The Walsh Filter (FWHT Attention)
- **Qué se probó:** Sustituir las convoluciones 3x3 por filtrado global en el dominio de Walsh: transformar la imagen con FWHT, aprender un "dial de ecualización" por frecuencia, e invertir con IFWHT.
- **Setup:** CIFAR-10, 3 bloques residuales de filtrado Walsh, 408,842 params, 50 épocas, AdamW + OneCycleLR, CPU.
- **Resultado principal:** **74.04%** (época 26; final 73.15%).
- **Hallazgo:** La FWHT es un extractor de características SOTA para visión. El filtrado global O(N log N) compite con arquitecturas densas. El techo del 74% indica necesidad de mayor profundidad jerárquica.

#### V36 — The Walsh-MNIST MLP (Zero-Weight)
- **Qué se probó:** Resolver MNIST sin matriz de pesos densa: transformar a dominio Walsh, modular el espectro con 128 neuronas de atención (ecualizador de 1024 frecuencias), calcular energía media y clasificar con un lineal 128→10.
- **Setup:** MNIST (padded 32x32), 263,690 params, 10 épocas, AdamW, CPU.
- **Resultado principal:** **98.54%** (época 9; 92.9% en época 1).
- **Hallazgo:** Walsh es el "lenguaje de los trazos" de MNIST. El modelo "Zero-Weight" (solo diales de atención sobre frecuencias fijas) alcanza nivel SOTA. Base ideal para Walsh-Transformers de contexto infinito.

#### V37 — Seismic Walsh Optimizer
- **Qué se probó:** Fusionar Seismic Descent (deformación del paisaje de pérdida) con la Transformada de Walsh: el optimizador genera "energía sísmica" en dominio Walsh y la proyecta como vibración estructurada sobre los pesos, con amplitud senoidal.
- **Setup:** MNIST, MLP básico, 10 épocas, amplitud sísmica ±0.01.
- **Resultado principal:** 96.30% (época 3); final 94.89%. V38 (con cooling + LR refinado): **97.25%**.
- **Hallazgo:** El "Seismic Cooling" es vital: la energía inicial explora, la calma final asienta. La Walsh es más efectiva como "gafas" (V36: 98.5%) que como "vibrador" del suelo (V37: 97.2%).

#### V39 — The Banded Walsh Equalizer
- **Qué se probó:** Compresión extrema: agrupar las 1024 frecuencias de Walsh en 4 bandas (graves/medios/agudos) y aprender solo 4 params multiplicativos + 4 aditivos por neurona (núcleo de 512 params).
- **Setup:** MNIST (padded 32x32), núcleo de atención 512 params, 786,954 totales (dominado por FC final), 10 épocas, AdamW + OneCycleLR.
- **Resultado principal:** **93.98%** (época 10). V39b con LR=0.0001: **96.00%**.
- **Hallazgo:** La información esencial de una imagen puede comprimirse masivamente agrupando bandas de frecuencia ortogonales. El cuello de botella pasa a ser la capa densa final.

#### V40 — The Nano-Walsh Net
- **Qué se probó:** El límite absoluto de compresión: 3 capas de filtrado Walsh (128 bandas), Average Pooling extremo 8x8 (a 4x4) y clasificador enano (16→10).
- **Setup:** MNIST (padded 32x32), **938 params totales**, 15 épocas, AdamW + OneCycleLR, CPU.
- **Resultado principal:** **92.12%** (V40b con 256 bandas: **93.50%** con 1,706 params).
- **Hallazgo:** Récord de eficiencia (Accuracy per Parameter). Con 8x menos params que una regresión logística (7,850), iguala su rendimiento. El agrupamiento excesivo de bandas es el factor limitante.

#### V41–V48 — La Frontera de las Proyecciones Congeladas
- **Qué se probó:** Exploración exhaustiva de la capa de entrada congelada en MNIST: ruido blanco global, base de Fourier, parches aleatorios, contraste diferencial (+1/-1), on-center/off-surround, Perlin local.
- **Setup:** MNIST, capa de entrada congelada + readout MLP entrenable, 10-100 épocas.
- **Resultado principal:** Mejor configuración estructural V48h (Quad Contrast): **98.44%**; récord absoluto V46 (Local Perlin + MLP 512): **98.58%**; mejor Acc/Param V44 (parches 6x6-14x14): 97.90% con 20,490 params.
- **Hallazgo:** La localidad esparcida y el contraste diferencial son los dos pilares. El contraste (+/-) es más informativo que la suma. La primera capa congelada es "inflexible" ante rotaciones, limitando el techo a ~98.5%.

#### V50 — Stroke Neurons (Curvas de Bézier)
- **Qué se probó:** Sustituir los pesos densos de píxeles por geometría vectorial procedural: cada neurona "dibuja" una curva de Bézier cuadrática (3 puntos de control + grosor) con contraste on-center/off-surround.
- **Setup:** MNIST, 256 neuronas, 8 params/neurona (1,536 en capa 1), 35,722 params totales, 14 épocas.
- **Resultado principal:** **97.88%**.
- **Hallazgo:** 100% caja blanca: la red aprende un "alfabeto visual" (arcos para 0/8/2, líneas para 1/7/4). Compresión ~99.2% de la capa de entrada. Invarianza a resolución y robustez a ruido de alta frecuencia.

#### V51 — Matchstick Neurons (Line Segments)
- **Qué se probó:** Simplificar las Stroke Neurons eliminando la curvatura: usar solo segmentos de línea recta definidos por 2 puntos + 2 grosores (6 params/neurona).
- **Setup:** MNIST, 256 neuronas, 6 params/neurona, 10 épocas, LR bajo (0.005 → optimizado).
- **Resultado principal:** **98.30%** (época 9; final 98.17%).
- **Hallazgo:** La línea recta es el "átomo" de la forma para MNIST. Los parámetros de coordenadas son extremadamente sensibles al LR; un LR más bajo desbloquea rendimiento superior. Compresión ~130x vs capa densa.

#### V52 — Double Matchstick Neurons
- **Qué se probó:** Permitir dos segmentos de línea por neurona (10 params/neurona) para detectar estructuras compuestas (esquinas, cruces) mediante distancia mínima (OR espacial).
- **Setup:** MNIST, 10 params/neurona, 10 épocas.
- **Resultado principal:** **97.52%** (vs 97.78% de v51 con 1 línea).
- **Hallazgo:** Ley de rendimientos decrecientes: para MNIST es más eficiente tener muchos detectores simples que detectores complejos. Las dos líneas tienden a colapsar o competir redundantemente.

#### V54 — RGB Matchstick Neurons (CIFAR-10)
- **Qué se probó:** Trasladar las Matchsticks a CIFAR-10 añadiendo sensibilidad al color RGB (vector de pesos aprendible por neurona).
- **Setup:** CIFAR-10 (32x32 RGB), 512 neuronas, 9 params/neurona (6 geometría + 3 color), 10 épocas.
- **Resultado principal:** **61.18%** (época 9; final 60.79%).
- **Hallazgo:** Arranque explosivo (>52% en época 1): la red localiza bordes y contrastes de color instantáneamente. Actúa como banco de filtros Gabor aprendibles pero más parsimoniosos.

#### V55 — Symmetry Break & Geometric Blindness
- **Qué se probó:** Inicializar las 256 neuronas como un segmento vertical idéntico en el centro (14,14) para ver si el gradiente dispersa las neuronas.
- **Setup:** MNIST, 100 épocas, inicialización central.
- **Resultado principal:** **95.71%** (época 97).
- **Hallazgo:** La simetría se rompe pero extremadamente lento (10x más épocas que v51 y peor resultado). La inicialización aleatoria actúa como "infraestructura de transporte de gradiente"; sin ella, la red pasa el 90% del tiempo buscando dónde están los datos.

#### V57 — Grid Initialization (Structured Prior)
- **Qué se probó:** Inicializar las 256 neuronas en una rejilla perfecta 16x16 cubriendo toda la imagen, cada una como pequeño segmento vertical.
- **Setup:** MNIST, 256 neuronas, 10 épocas, inicialización en rejilla.
- **Resultado principal:** **97.16%** (época 9); primera época **95.84%**.
- **Hallazgo:** La rejilla elimina la "ceguera geométrica" desde el primer batch (arranque explosivo) y es muy estable. Pero el "caos" de la inicialización aleatoria (v51: 98.30%) proporciona diversidad de búsqueda superior a largo plazo.

#### V59 — DCT Attention Neurons
- **Qué se probó:** Inspirado en JPEG: reemplazar los pesos densos por modulación DCT diferenciable. Cada neurona aprende un kernel KxK de coeficientes DCT (K=8, 64 params) aplicado al cuadrante de baja frecuencia.
- **Setup:** MNIST (28x28), 512 neuronas ocultas, ~39k params totales, 15 épocas, compresión 12.25x.
- **Resultado principal:** **98.12%** (96% en época 1; 121.6s).
- **Hallazgo:** El sesgo de baja frecuencia regulariza contra ruido de píxel. Las neuronas sintetizan gradientes suaves, bordes Gabor y estructuras center-surround con solo 64 coeficientes. Mecanismo de campo receptivo global muy eficiente.

### Era DCT/JPEG (v60–v67)

> **Tema:** consolidación de la DCT como herramienta universal. Se pasa de comprimir la capa de entrada a comprimir **toda la topología interna** de MLPs y Transformers, y se descubre que el lenguaje es una "onda semántica" comprimible.
> **Hito clave:** V63 comprime un MLP 56.2x sin perder precisión; V66 demuestra que un LLM 100% DCT aprende; V65 revela que el significado del texto reside en las bajas frecuencias.

#### V60 — Extreme Compression (MNIST)
- **Qué se probó:** Llevar la compresión DCT al límite usando solo 4x4 coeficientes DCT por neurona (16 params/neurona, 49x compresión vs 784 densos).
- **Setup:** MNIST, 16 pesos aprendibles por neurona.
- **Resultado principal:** **93.17%**.
- **Hallazgo:** La mayor parte de la información semántica de MNIST reside en los primeros 16 componentes de baja frecuencia; las altas frecuencias son mayormente redundantes.

#### V61 — Global DCT Attention (CIFAR-10)
- **Qué se probó:** Aplicar modulación de frecuencia global a imágenes color 3 canales (32x32) en el dominio DCT.
- **Setup:** CIFAR-10, modulación global de frecuencia.
- **Resultado principal:** **62.64%**.
- **Hallazgo:** Capturar estructura global en el espacio de frecuencias es significativamente más efectivo que MLPs en espacio de píxeles, proporcionando campo receptivo global desde la primera capa.

#### V62 — Convolutional DCT Kernels (CIFAR-10)
- **Qué se probó:** Sintetizar kernels convolucionales locales 8x8 a partir de un conjunto reducido de coeficientes DCT 4x4.
- **Setup:** CIFAR-10, kernels 8x8 sintetizados desde DCT 4x4.
- **Resultado principal:** **72.72%** (época 1: 62.79%, convergencia instantánea).
- **Hallazgo:** La modulación de frecuencia local combina la invariancia espacial de las CNNs con el sesgo inductivo de compresión tipo JPEG. Los filtros son inherentemente suaves y biológicamente plausibles.

#### V63 — The All-DCT MLP
- **Qué se probó:** Comprimir toda la topología interna de un MLP proyectando todas las capas ocultas al dominio de frecuencias (cada `nn.Linear` reemplazado por `DCTLinear`).
- **Setup:** MNIST, MLP 3 capas (784→512→512→10), núcleo 64x64 por capa oculta.
- **Resultado principal:** **97.59%** con 11,914 params (vs 669,706 densos, **56.2x compresión**).
- **Hallazgo:** Las representaciones semánticas internas de una red son altamente comprimibles en el dominio de frecuencias. El enrutamiento de conceptos "baja frecuencia" de capa a capa previene matemáticamente la propagación de ruido de alta frecuencia.

#### V64 — The DCT-Transformer (NLP)
- **Qué se probó:** Aplicar la compresión DCT a los FFNs SwiGLU de un Transformer para modelado de lenguaje autorregresivo.
- **Setup:** Transformer 4 capas, 4 cabezas, d_model=128, dataset tiny-thinker (vocab 16384), FFN DCT 32x64.
- **Resultado principal:** Loss final **6.0820** (desde 9.8707); compresión FFN **32.0x** (24,576 vs 786,432 params).
- **Hallazgo:** El concepto de "frecuencia" aplica al lenguaje: conceptos amplios (part-of-speech, significado central) residen en bajas frecuencias. Los FFNs de LLMs son extremadamente redundantes.

#### V65 — The "JPEG" of Language (Text DCT)
- **Qué se probó:** Exploración teórica: aplicar DCT a lo largo de la dimensión de secuencia de embeddings y truncar las altas frecuencias (estilo JPEG) para ver qué queda.
- **Setup:** Embeddings preentrenados de tiny-thinker (V=16384, d=512), frase de 25 tokens, truncación al 100/50/25/15%.
- **Resultado principal:** 50% → pierde "brave" (se vuelve "The"); 25% → "The The The decided to to to..." (bloques semánticos); 15% → solo el DC (promedio semántico).
- **Hallazgo:** La gramática y el significado fundamental de una frase residen en las **bajas frecuencias**. La generación autorregresiva token-a-token es ineficiente; una arquitectura coarse-to-fine (generar la onda semántica primero, luego los detalles) sería más óptima.

#### V66 — The Fully-JPEG LLM (100% DCT Compression)
- **Qué se probó:** Comprimir el 100% de la topología interna de un Transformer: no solo los FFNs, sino también las proyecciones de Atención (Q, K, V, O) con DCTLinear.
- **Setup:** Transformer 4 capas, 4 cabezas, d_model=128, núcleos DCT 32x32 para atención (16x) y 32x64 para FFN (32x).
- **Resultado principal:** Loss final **6.2214** (desde 9.8772); convergencia suave y estable.
- **Hallazgo:** La atención es armónica: comprimir Q y K fuerza a buscar relaciones con ondas suaves en lugar de ruido token-a-token. Al tener pocos parámetros de baja frecuencia, la red no puede sobreajustar al ruido de entrenamiento.

#### V67 — The Hybrid Spectral GPT (DCT + Walsh)
- **Qué se probó:** Mezclar dominios espectrales según el propósito cognitivo: DCT para Atención (semántica continua) y FWHT (Walsh) para FFN (lógica binaria y reglas).
- **Setup:** Transformer 4 capas, 4 cabezas, d_model=128, atención DCT 16x, FFN Walsh 32x, dataset tiny-thinker.
- **Resultado principal:** Loss final **6.3141** (desde 9.8490); 2,270,336 params totales (incluye 2M de embeddings).
- **Hallazgo:** La "interlingua" espectral funciona: la red enruta conceptos semánticos continuos (DCT) hacia procesadores lógicos afilados (Walsh) y viceversa. Los FFNs Walsh podrían implementarse solo con sumas/restas en hardware especializado.

### Era Espectral & PAC (v69–v89)

> **Tema:** consolidación de la librería `attention_neuron`, interpretabilidad espectral, el algoritmo PAC (Purifying Archetype Classifier) y el salto a la **mega-capa espectral 16K** y la **memoria holográfica O(1)**.
> **Hito clave:** V87 rompe la ley de escalado cuadrático (65,540x compresión); V88 demuestra memoria holográfica O(1) con recall exitoso; V76 logra 93.50% con 280 arquetipos interpretables.

#### V69 — Spectral Interpretability & Modularization
- **Qué se probó:** Modularizar los prototipos en la librería `attention_neuron` (AttentionLinear, RosettaLinear, DCTLinear, WalshLinear) y visualizar qué aprenden las neuronas DCT 2D en MNIST.
- **Setup:** MNIST, clasificador 1 capa (784→10) con núcleo DCT-2D 16x16, 2,570 params.
- **Resultado principal:** **92.4%** (cerca del límite teórico lineal).
- **Hallazgo:** La DCT fuerza a aprender "formas globales" en lugar de memorizar ruido de píxel. Cada neurona sintetiza un "ideal platónico" del dígito (Neuron 0: anillo continuo; Neuron 1: barra vertical). La DCT 2D produce plantillas legibles por humanos.

#### V70 — Hierarchical Composition & Visual Atoms
- **Qué se probó:** Arquitectura jerárquica de 2 capas: 20 neuronas ocultas DCT-2D (12x12) como "átomos visuales" + mezclador lineal estándar.
- **Setup:** MNIST, ~3,000 params (200x compresión), ReLU.
- **Resultado principal:** **95.96%**.
- **Hallazgo:** El cuello de botella de 20 neuronas fuerza una representación por partes: la capa 1 desarrolla detectores de trazos (verticales, horizontales, bucles, intersecciones) y la capa 2 los compone (el 8 = bucles superior+inferior; el 1 = barra vertical central). Emergencia de representación basada en partes.

#### V71 — The Neural Vectorizer (SGD Image Tracing)
- **Qué se probó:** Invertir el proceso: optimizar 12 Stroke Neurons (curvas Bézier) con SGD para "dibujar" una imagen MNIST específica (un '5').
- **Setup:** MNIST, 12 trazos Bézier (3 puntos de control + opacidad), Adam, MSE, <60 épocas.
- **Resultado principal:** Convergencia casi perfecta en segundos.
- **Hallazgo:** Prueba de concepto para un **modelo generativo vectorial**: predecir Nx3 coordenadas de trazos en lugar de píxeles. Ventajas: huella diminuta (72 números vs 784 píxeles), escalabilidad infinita (SVG), estructura limpia.

#### V74 — Archetype Nearest Centroid Classification
- **Qué se probó:** Clasificador de cero parámetros entrenables: medir distancia MSE entre la imagen y 10 arquetipos (promedios de clase).
- **Setup:** MNIST test (10,000 imágenes), arquetipos de píxel y vectoriales (15 trazos Bézier).
- **Resultado principal:** **82.03%** (píxel) / **78.66%** (vector).
- **Hallazgo:** La forma topológica global explica la mayor parte de la varianza de MNIST sin entrenar. Los arquetipos vectoriales son más interpretables pero pierden artefactos estadísticos que ayudan al MSE a distinguir casos difíciles.

#### V75 & V76 — The Purifying Archetype Classifier
- **Qué se probó:** Algoritmo PAC: clustering supervisado dinámico que aísla errores y purifica los arquetipos base (V75: sub-arquetipos de errores; V76: recálculo del base sin errores + reasignación tipo K-Means).
- **Setup:** MNIST, 60 sub-arquetipos (V75) → 280 arquetipos puros (V76).
- **Resultado principal:** 86.9% (V75) → **93.50%** (V76).
- **Hallazgo:** Interpretabilidad absoluta: el conocimiento se almacena como 280 imágenes legibles. Si el modelo falla, se puede ver exactamente con qué arquetipo confundió la imagen. Reduce de 60,000 referencias (KNN) a 280.

#### V79 — Active Morphing Classifier (Inferencia Activa)
- **Qué se probó:** Análisis por síntesis: durante la inferencia, deformar activamente 10 arquetipos vectoriales (Bézier) con descenso de gradiente para encajar en la imagen de test; la clase con menor "esfuerzo elástico" gana.
- **Setup:** MNIST, 10 arquetipos base, 30 pasos Adam (lr=0.2), pérdida elástica λ=0.05, 50 imágenes de test.
- **Resultado principal:** **86.00%** sin entrenamiento tradicional.
- **Hallazgo:** El "pensamiento lento" (System 2) interpretable funciona. Errores principales: 4→9 y 7→9 (un '9' es un '4' cerrado; el muelle elástico no es suficientemente fuerte). Coste: ~36s para 50 imágenes.

#### V81 — PAC + K-NN Voting
- **Qué se probó:** En lugar de 1-NN sobre 491 arquetipos purificados, probar votación Top-K (K=1,3,5,10,15).
- **Setup:** MNIST, 491 arquetipos, distancia L2, 10,000 imágenes test.
- **Resultado principal:** 1-NN: **94.43%**; 3-NN: 93.23%; 5-NN: 92.48%; 10-NN: 91.48%; 15-NN: 90.41%.
- **Hallazgo:** Degradación estricta al subir K: los arquetipos son centros de masa resumidos, no datos crudos. La votación K-NN sufre "secuestro de vecindario" (clases con más sub-clústeres dominan). Para diccionarios ontológicos, **1-NN es matemáticamente correcto**.

#### V82 — DCT Image Reconstruction
- **Qué se probó:** Usar una "neurona DCT" generativamente: optimizar 64 coeficientes DCT (8x8) para reconstruir una imagen MNIST específica.
- **Setup:** MNIST, 64 coeficientes aprendibles, Adam (lr=0.1), 500 épocas, MSE.
- **Resultado principal:** MSE final **~0.0347**.
- **Hallazgo:** Los coeficientes DCT sirven como "receta" eficiente para generar imágenes. La neurona actúa como filtro paso-bajo aprendido que "sabe" representar dígitos.

#### V84 — Spectral Basis Comparison (DCT vs Walsh)
- **Qué se probó:** Comparar DCT vs Walsh-Hadamard para reconstrucción de imágenes en el framework Attention Neuron.
- **Setup:** MNIST, K=8 (DCT) vs K=16/32 (Walsh).
- **Resultado principal:** DCT captura la esencia con K=8; Walsh requiere K=16/32 para suprimir artefactos "blocky".
- **Hallazgo:** DCT gana en fidelidad de imagen; Walsh ofrece ventaja de hardware (solo sumas/restas, sin cosenos). Walsh requiere reordenamiento por sequency para ser útil en imágenes.

#### V85 & V86 — The Neural-PAC Prototypes
- **Qué se probó:** Integrar PAC en una red neuronal diferenciable usando DCT neurons: V85 (aprender a "dibujar" los dígitos con backprop selectivo positivo) y V86 (neurogénesis dinámica: crear nuevas neuronas para estilos desconocidos).
- **Setup:** MNIST, 10 neuronas DCT iniciales → crecimiento orgánico hasta 50/200, [SPAWN] al detectar error.
- **Resultado principal:** Taxonomía de estilos (un "4" se divide en "4 abierto", "4 cerrado", "4 inclinado").
- **Hallazgo:** Neurogénesis por reconstrucción (inicializar nuevas neuronas desde imágenes de error con DCT forward). Winner-Take-All: solo el arquetipo más cercano se actualiza, evitando interferencia entre estilos.

#### V87 — The 16K Mega-Layer Breakthrough
- **Qué se probó:** Benchmark de una capa 16,384x16,384 (típica de LLMs) comparando Dense vs Síntesis Espectral FWHT (K=64).
- **Setup:** CPU, capa 16,384x16,384, núcleo espectral 64x64.
- **Resultado principal:** **65,540x compresión** (4,096 vs 268M params), **65,540x reducción de memoria** (16KB vs 1GB), **40.2x más rápido** (0.0098s vs 0.3941s).
- **Hallazgo:** Muerte del "Memory Wall": la capa espectral cabe en caché L1/L2. Complejidad O(N log N) en lugar de O(N²). Camino viable a AGI hyperscale en dispositivos de consumo.

#### V87b — Mega-Layer Learning Validation
- **Qué se probó:** Validar que la mega-capa espectral realmente aprende mejor que baselines iso-paramétricos (4,096 params) en señales de 16,384 dimensiones con decaimiento espectral.
- **Setup:** 512 señales de 16,384 dims, α=2.0, Adam lr=1e-2, 40 épocas.
- **Resultado principal:** **MSE 8.95e-10** (vs 3.74e-9 de baselines, **4.18x menor**); PEI 3.09e8.
- **Hallazgo:** La anchura sintetizada es real: cada frecuencia Walsh es una base global que actualiza los 16,384 elementos de salida. No es un truco de compresión vacío. Etiquetado [SEÑAL].

#### V87c — Cross-Basis Verification
- **Qué se probó:** Dirimir si la ventaja de FWHT era un artefacto de base coincidente (datos generados en Walsh) o universal, probando en bases Walsh, DCT y aleatoria + baseline PCA.
- **Setup:** 3 bases de generación de datos, 4 modelos iso-paramétricos, 512 señales.
- **Resultado principal:** FWHT: 8.95e-10 (Walsh), 2.48e-5 (DCT), 5.92e-5 (aleatoria). PCA: 4.84e-7 (todas las no nativas).
- **Hallazgo:** **[ANCLA-NEGATIVO]**: FWHT es un **prior estructural** eficiente para señales espectralmente alineadas, no un expansor universal. Elegir el subespacio de varianza dominante (PCA) es la variable fundamental.

#### V87d — Smooth Walsh vs DCT vs Blocky Walsh
- **Qué se probó:** Comparar DCT, Walsh blocky, Walsh suave (low-pass espectral y interpolación bilineal 2D) en señales continuas (decaimiento DCT) vs discretas (decaimiento Walsh).
- **Setup:** 5 arquitecturas, 512 señales de 16,384 dims, iso-paramétrico 4,096.
- **Resultado principal:** DCT: 5.70e-7 (continuas, 43.5x mejor que Walsh); Walsh: 8.95e-10 (discretas, 1.71x mejor que DCT).
- **Hallazgo:** **DCT es el prior óptimo para señales continuas** (visión, habla, embeddings); **Walsh para señales discretas** (lógica, árboles, estados). El low-pass espectral no convierte una onda cuadrada en coseno.

#### V88 — El Hipocampo Holográfico (Memoria O(1))
- **Qué se probó:** Romper el límite de la ventana de contexto: memoria holográfica con FWHT temporal que comprime 51,200 tokens en 64KB constantes, con "olvido selectivo" (truncar altas frecuencias) e interferencia holográfica.
- **Setup:** Streaming de 51,200 tokens, embedding D=256, chunk 512, K_micro=64, capacidad total 64KB, supresión de sesgo de consulta + saliencia de amígdala (x150).
- **Resultado principal:** **[ÉXITO MASIVO]**: similitud coseno con el target **0.4861** (vs 0.0081 de ruido control) tras 51,200 tokens de distracción.
- **Hallazgo:** La memoria holográfica O(1) funciona. Base para RAG endógeno sin bases de datos vectoriales externas y LLMs de contexto verdaderamente infinito.

#### V89 — El Cerebelo Espectral (Inferencia Dinámica)
- **Qué se probó:** Cognición dual: vía rápida (Cerebelo FWHT, 10,240 params) + vía lenta (Córtex MLP, >900k params), con enrutamiento por entropía predictiva (early-exit si entropía < 0.5).
- **Setup:** MNIST, 3 épocas, batch size 1, umbral de entropía 0.5.
- **Resultado principal:** El Cerebelo absorbió **93.7%** del trabajo (91.95% precisión); el Córtex solo 6.3% (88.78%). Global: 91.75%. Velocidad: 0.245ms vs 0.542ms (**2.2x más rápido**).
- **Hallazgo:** La duda del Cerebelo está perfectamente calibrada: los casos que rechaza son genuinamente los más difíciles. Inferencia dinámica validada para Edge Computing.

### Era Geométrico/Foveación (v90–v99)

> **Tema:** neuronas "analógicas" con agregadores múltiples, foveación (espiral, log-polar), invariancia RST (Fourier-Mellin) y neuronas geométricas ultra-comprimidas (triangulares).
> **Hito clave:** V90e (Resonador Holográfico) logra 97.92% MNIST; V97 demuestra +14.86% de robustez con Fourier-Mellin; V99c alcanza 84.21% con solo ~11k params.

#### V90c/d/e — Evolución de la Placa Analógica
- **Qué se probó:** Tres evoluciones de la "Placa Analógica": V90c (gating dinámico sigmoid para elegir agregador por entrada), V90d (máscara espectral aprendible por neurona) y V90e (interferencia compleja en dominio Walsh con componentes reales/imaginarios).
- **Setup:** MNIST, 64 neuronas, agregadores SUM/VAR/L2/LSE/WALSH.
- **Resultado principal:** V90c: 97.26%; V90d: 97.27%; **V90e: 97.92%** (récord).
- **Hallazgo:** La especialización dinámica supera a la mezcla estática. La resonancia holográfica (interferencia compleja) es una representación mucho más densa que la suma lineal, validando la "Memoria Holográfica" para clasificación.

#### V90b — Placa Analógica Adaptativa
- **Qué se probó:** Permitir que cada neurona aprenda su propia mezcla óptima de agregadores (SUM, VAR, L2, LSE, WALSH_ENERGY) mediante `mixture_logits` con Softmax.
- **Setup:** MNIST, 64 neuronas adaptativas, clasificador lineal 64→10.
- **Resultado principal:** Pendiente de ejecución (solo diseño e implementación).
- **Hallazgo:** Hipótesis: la diversidad matemática (especialmente WALSH_ENERGY y VAR) es útil para el aprendizaje de características. El tensor intermedio (B, 64, 784) es eficiente pero podría ser cuello de botella en capas grandes.

#### V93 — Spiral Pixel Ordering (Foveated Attention)
- **Qué se probó:** Serializar los píxeles en orden espiral centro-fuera (foveación biológica) en lugar del raster estándar, procesado por una arquitectura Walsh.
- **Setup:** MNIST, 3 épocas, hidden_dim=32, FWHT.
- **Resultado principal:** **22.97%** vs 22.12% raster (**+0.85%**).
- **Hallazgo:** El raster empieza más fuerte pero se estanca; el espiral muestra gradiente de aprendizaje positivo sostenido. La foveación actúa como sesgo inductivo natural para datasets centrados. La FWHT se alinea mejor con el raster (arranque más lento del espiral).

#### V93b — Fractal Hierarchical MLP
- **Qué se probó:** Proporcionar a la red una representación multiresolución: vector de 1365 dims con promedios globales, de cuadrante y sub-cuadrante hasta nivel de píxel.
- **Setup:** MNIST, 5 épocas, hidden_dim=256.
- **Resultado principal:** Época 1: 97.08% (+0.32% vs raster); Época 3: 97.90% (+0.22%); Final: 97.72% (-0.39%).
- **Hallazgo:** La entrada jerárquica acelera la convergencia inicial (bypass del pooling básico), pero el MLP raster alcanza y supera al final. Útil para "Fast Thinking" o few-shot.

#### V95 — Log-Polar Spiral Sampling (Analog Foveation)
- **Qué se probó:** Muestreo continuo log-polar con espiral logarítmica y muestreo gaussiano multiescala concentrado en el centro, transformando la imagen en un flujo 1D de 1024 valores.
- **Setup:** MNIST, 5 épocas, MLP estándar, hidden_dim=256, `grid_sample` bilineal.
- **Resultado principal:** **98.29%** (época 5) vs 97.96% raster (**+0.33%**).
- **Hallazgo:** El muestreo log-polar sufre una penalización inicial (blur por interpolación) pero supera al raster al final. Prioriza la fóvea e ignora las esquinas vacías, asignando más "bandwidth" a los trazos centrales.

#### V97 — Fourier-Mellin Invariance (Torture Test)
- **Qué se probó:** Usar la Transformada Fourier-Mellin como preprocesado para lograr invariancia RST (rotación, escala, traslación) en un "torture test" de MNIST (rotación hasta 90°, desplazamiento hasta 20%).
- **Setup:** MNIST torture test, 10 épocas, MLP estándar, hidden_dim=256.
- **Resultado principal:** **35.20%** vs 20.34% raster (**+14.86%**).
- **Hallazgo:** El MLP raster falla casi por completo (~20%) al depender de posiciones de píxel. Fourier-Mellin casi duplica la robustez. El límite es la pérdida de fase (magnitud FFT), que sacrifica detalles finos.

#### V98 — Invariant Spectral Attention (ISA Hybrid)
- **Qué se probó:** Hibridar Fourier-Mellin (V97) con Atención Espectral Walsh para filtrar la firma invariante en el dominio de frecuencias.
- **Setup:** MNIST torture test (90° rotación + 20% shift), 10 épocas.
- **Resultado principal:** Pico **41.43%** (+2.12% vs V97); final 37.72%.
- **Hallazgo:** El filtrado en dominio Walsh es superior al MLP para interpretar firmas invariantes. Imita el pipeline visual humano: foveación/invarianza (FM) + procesamiento frecuencial jerárquico (atención espectral).

#### V98b — DCT vs Walsh en ISA
- **Qué se probó:** Comparar Walsh (binario, rígido) vs DCT (sinusoidal, suave) como mecanismo de atención espectral sobre la firma Fourier-Mellin.
- **Setup:** MNIST torture test, 10 épocas.
- **Resultado principal:** DCT pico **41.93%** (+0.18% vs Walsh); Walsh final 40.97% (+1.38% vs DCT).
- **Hallazgo:** DCT alcanza el pico más alto (base sinusoidal más "natural" para FM), pero Walsh es más robusto al final (la naturaleza binaria actúa como regularización implícita contra ruido espectral fino).

#### V99 — Triangular Attention Neuron (1D)
- **Qué se probó:** Neuronas ultra-comprimidas definidas por solo 2 parámetros (centro y ancho) que generan una máscara triangular 1D, reemplazando matrices densas.
- **Setup:** MNIST, 1024 neuronas triangulares (2,048 params) + 512 (1,024) + 10 densas (5,120); total ~8,192 params (90%+ reducción).
- **Resultado principal:** **79.59%** (raster) vs 74.72% (espiral).
- **Hallazgo:** ~80% con solo 11k params es un resultado fuerte. El raster supera al espiral (MNIST es "line-oriented"). Inestabilidad observada (dips a 10-30%) por interacción entre ancho y BatchNorm.

#### V99b/c — Multi/Omni-View Triangular Attention
- **Qué se probó:** Concatenar múltiples vistas de la misma imagen: V99b (3 vistas: raster, transpuesta, espiral) y V99c (5 vistas: + diagonales ±45°).
- **Setup:** MNIST, 2352 inputs (3 vistas) / 3920 (5 vistas), ~11k params constantes.
- **Resultado principal:** V99b: pico 78.21%; **V99c: pico 84.21%** (final 75.89%).
- **Hallazgo:** La integración multi-vista es superior bajo restricción paramétrica extrema. Las diagonales juntas suman 36% de la atención (más informativas que columnas). Inestabilidad numérica ("cliff" a 11.35%) cuando los anchos se encogen.

### Era Cone/Haar (v101–v109)

> **Tema:** neuronas cónicas 2D (4 params) para visión, conos temporales para lenguaje (ConeAttn), wavelets de Haar y el descubrimiento de que el FFN está masivamente sobreparametrizado (NarrowFFN).
> **Hito clave:** V101 logra 94.30% MNIST con 3,850 params; V103 descubre que los radios de los conos crecen con la profundidad (jerarquía V1→V4); V105 demuestra que NarrowFFN captura el 99% del FFN denso con 11.5x menos params.

#### V101 — Cone Attention Neurons (Eficiencia Extrema)
- **Qué se probó:** "Neurona de Atención Cónica 2D" con solo 4 parámetros (Cx, Cy, Radio, Amplitud) que define un campo receptivo cónico, imitando células ganglionares de la retina. Con amplitudes negativas (inhibición) y LR bajo.
- **Setup:** MNIST, 3,850 params, LR=0.001, amplitudes inicializadas entre -1.0 y 1.0.
- **Resultado principal:** **94.30%** (90.44% en época 1).
- **Hallazgo:** Reducir el espacio de búsqueda a 4 dims/neurona hace el paisaje de pérdida liso y directo. La inhibición (amplitudes negativas) es crucial. V102 (salida triangular 1D) reveló auto-organización topológica: la red crea un espacio latente estructurado donde características similares están físicamente juntas.

#### V103 — Cone Neurons for Language Modeling
- **Qué se probó:** Conos temporales en un Transformer: ConeAttn (conos como mezcla temporal), ConeFFN (conos en FFN) y FullCone (todo cónico).
- **Setup:** d_model=64, Transformer pequeño, 20 épocas.
- **Resultado principal:** ConeAttn: +4.0% loss con 24% menos params; ConeFFN: +5.2% con 30% menos; FullCone: +15.1% con 53% menos.
- **Hallazgo:** **Los radios de ConeAttn crecen con la profundidad** (L0: [3.0, 9.0] → L2: [4.1, 10.3]): jerarquía V1→V4→IT emergiendo en lenguaje. ConeFFN colapsa a radio ~1 (cada neurona lee 1 dim). Proyección a escala: atención 32x menos params, O(N) en contexto, sin KV-cache.

#### V103–V106 — The Haar Wavelet Era
- **Qué se probó:** Wavelets de Haar como base de representación: V103 (grid fijo), V104 (selección rank=2), V105 (embudo a 3 neuronas), V106 (selección rank=8 + BatchNorm Espectral).
- **Setup:** MNIST, 2.6k–12.6k params.
- **Resultado principal:** V103: 93.08%; V104: 78.10% (fallo); V105: 39.67% (fallo); **V106: 96.20%** (94.77% en época 1).
- **Hallazgo:** Haar sabe *dónde* está el trazo (localización espacial) y detecta orientaciones por separado. El BatchNorm Espectral fue el "héroe silencioso" equilibrando energía entre escalas. Rank=2 es insuficiente; Rank=8 es el punto dulce.

#### V104 — ConeFFN Radius Collapse
- **Qué se probó:** Investigar por qué los conos del FFN colapsaron a radio ~1 en V103, probando formas triangular vs gaussiana y floors de radio.
- **Setup:** d_model=64, Transformer, ConeFFN con variantes.
- **Resultado principal:** Todos los conos colapsan a radio ~0.9-1.0; forzar floor=4 empeora (+8.1%).
- **Hallazgo:** Con d_model=64 no hay redundancia entre dimensiones vecinas; cada dim ya codifica información independiente. El ConeFFN degenera en una matriz sparse accidental (pick-one). Para que la topología emerja se necesita d_model >> 64.

#### V105 — Is FFN Just a Dimension Gate?
- **Qué se probó:** Comparar DenseFFN (d→4d→d), NarrowFFN (d→d+GELU), BottleneckFFN (d→d/4→d) y DimGate (x*sigmoid(g)) en un Transformer.
- **Setup:** d_model=128, atención causal estándar, 20 épocas.
- **Resultado principal:** Dense: 1.5527; **NarrowFFN: 1.5689 (+1.0% con 11.5x menos params)**; Bottleneck: +2.7%; DimGate: +5.9%.
- **Hallazgo:** El FFN necesita recombinación lineal pero NO expansión a 4d. La expansión estándar es un derroche masivo. Proyección a LLaMA-7B: NarrowFFN reduciría el modelo de 7B a 3.2B (54% menos).

#### V106 — ConeAttn + NarrowFFN Combined
- **Qué se probó:** Combinar las dos victorias independientes (ConeAttn + NarrowFFN) en un solo modelo.
- **Setup:** d_model=128, Transformer.
- **Resultado principal:** Cone+Narrow: +10.6% (vs +2% esperado aditivo); PEI 0.1173 > baseline 0.1113.
- **Hallazgo:** Los ahorros NO se suman linealmente (interacción negativa). El modelo necesita al menos UN componente fuerte. Pero el PEI mayor sugiere que al escalar d_model la brecha se cerraría. Recomendación: elegir UNA de las dos optimizaciones según el cuello de botella.

#### V107 — Feature Fusion (MNIST)
- **Qué se probó:** Evaluar características morfológicas (Island Signatures: componentes conectados por fila/columna) e intensidad (sumas globales y por fila/columna) en un MLP de 2 capas.
- **Setup:** MNIST, MLP 128 hidden, representaciones de 57D a 897D.
- **Resultado principal:** Baseline píxeles: 97.75%; Intensity (57D): 90.26%; Islands (56D): 87.35%; **Intensity+Islands (113D): 94.70%**; Full Fusion (897D): 97.51%.
- **Hallazgo:** Las características compactas (113D) logran 94.7% con 13.7x reducción dimensional. El MLP ya extrae estas características de los píxeles (Full Fusion = Baseline). Ideal para TinyML.

#### V107 — Iso-Budget Comparisons
- **Qué se probó:** Comparar arquitecturas (baseline, NarrowFFN, DimGate) con presupuesto fijo de parámetros (~158K y ~612K) variando capas y d_model.
- **Setup:** 2 budgets, configs de L=3 a L=30, d=64 a 256.
- **Resultado principal:** Baseline siempre gana; DimGate empeora drásticamente con profundidad (L=20: 1.9720 vs L=3: 1.5768).
- **Hallazgo:** **DimGate es matemáticamente colapsable**: L capas de x*sigmoid(g) ≡ 1 capa con gate acumulado. La profundidad no crea representaciones nuevas. La receta "pocas capas, mayor d" gana. DimGate solo sirve como gate auxiliar.

#### V108 — nGPT + ConeAttn
- **Qué se probó:** Combinar nGPT (normalización en hiperesfera) con ConeAttn y variantes.
- **Setup:** d_model=128, 20 épocas, lr=3e-3, alpha_init=0.05.
- **Resultado principal:** Standard: 1.5518; nGPT+Cone+Dense: 1.8586 (+2.3% mejor que nGPT puro 1.9024); nGPT+Cone+Narrow: 1.9439.
- **Hallazgo:** nGPT converge MÁS LENTO que el Transformer estándar con hiperparámetros estándar (necesita lr=1e-2+). ConeAttn mejora nGPT incluso en la hiperesfera. DimGate es identidad exacta en S^(d-1) (la normalización cancela su único efecto). Nota: la demostración algebraica de DimGate en el doc tiene una patología de inicialización (g=0 uniforme), no una imposibilidad algebraica.

#### V109 — Cross-Neuron and Representation Comparison
- **Qué se probó:** Comparación masiva (16 configs) de 4 tipos de neurona (MLP, Triangular, DCT, Walsh) x 4 representaciones (Intensity, Islands, I+Is, Pixels) con 32 hidden units.
- **Setup:** MNIST, 32 hidden units, representaciones de 57D a 784D.
- **Resultado principal:** MLP+Pixels: 96.13%; Walsh+Pixels: 86.71%; **Triangular+Islands: 80.02% con solo 426 params**; MLP+I+Is: 91.85%.
- **Hallazgo:** Las neuronas especializadas son sensibles a la representación: Triangular+Islands es un breakthrough de eficiencia (426 params). Walsh supera a DCT en píxeles crudos (validando v87d: dígitos son casi binarios). Nota: la matriz no es iso-parámetro (426 vs 25k).

### Era Espectral/Holográfico (v110–v146)

> **Tema:** consolidación de los híbridos morfo-espectrales, neuronas Walsh suaves, optimizadores espectrales (SWO/ARSO), compresión espectral de LLMs (GPT-2), neuronas polimórficas con cerebelo espectral y la explosión de la **memoria holográfica** (131k ítems, zero-shot MNIST).
> **Hito clave:** V122 (Smooth Walsh) supera al Dense con 4x menos params; V129 descubre el umbral mágico del 50% en pruning de GPT-2; V137 ejecuta capas de 131,072 dims en GPU integrada; V146 logra 97.42% zero-shot.

#### V110 — Tri-Walsh Hybrid (Cerebro-Cerebelo)
- **Qué se probó:** Arquitectura híbrida de dos vías paralelas: Triangular (32 unidades) procesando Island Signatures (56D) + Walsh (32 unidades, k=16) procesando píxeles crudos (784D).
- **Setup:** MNIST, 1,290 params.
- **Resultado principal:** **93.03%** (~20x compresión vs baseline MLP 96.13%).
- **Hallazgo:** La sinergia entre el sesgo estructural local (Triangular) y el sesgo espectral global (Walsh) es la clave. Emparejar neuronas con su representación "natural" es más efectivo que capas densas más anchas.

#### V111 — Scaled Tri-Walsh Hybrid (H=96)
- **Qué se probó:** Escalar el híbrido v110 de 32 a 96 unidades ocultas por vía.
- **Setup:** MNIST, 3,850 params.
- **Resultado principal:** **94.20%** (+1.17pp vs v110).
- **Hallazgo:** Rendimientos decrecientes al triplicar params (+1.2%): el cuello de botella está en la representación o el núcleo espectral (k=16), no en el ancho. Estabilidad extrema (90% en época 3).

#### V112 — Spiral-Hybrid
- **Qué se probó:** Reemplazar los píxeles raster de la vía Walsh por muestreo espiral log-polar (1024 puntos), basado en los resultados positivos de v95.
- **Setup:** MNIST, 3,850 params.
- **Resultado principal:** **91.73%** (peor que v111: 94.20%).
- **Hallazgo:** En modelos ultra-compactos (<5k params), la red no tiene capacidad para "denoising" los artefactos de interpolación del espiral. El raster es más "Walsh-friendly" (la FWHT está diseñada para estructura rectangular discreta).

#### V113 — Full Morph-Spectral Hybrid
- **Qué se probó:** Fusionar todas las características (Islas + Intensidad + Píxeles) en un solo híbrido, con ancho mínimo de 0.02 para estabilizar las neuronas triangulares.
- **Setup:** MNIST, 5,386 params, k=32.
- **Resultado principal:** **93.01%** (test) — peor que v111 (94.20%).
- **Hallazgo:** "Menos es más" en ultra-compactos: añadir Intensidad no ayuda y aumenta params. La restricción de ancho mínimo eliminó la inestabilidad. k=16 ya captura el "low-pass" suficiente.

#### V117 — The Infinite Resolution Paradox
- **Qué se probó:** Aumentar la resolución de entrada de 4,096 a 32,768 muestras por imagen (espiral log-polar) manteniendo k=16 fijo.
- **Setup:** MNIST, ~2k params fijos, 3 épocas.
- **Resultado principal:** Degradación: 79.11% (4k) → 74.32% (32k) a 0°.
- **Hallazgo:** Resolución infinita requiere capacidad espectral infinita. Con k=16 fijo, cada coeficiente representa el promedio de 2,048 muestras (filtro paso-bajo extremo que destruye detalle). La saturación de redundancia crea "zonas muertas".

#### V118 — Spectral Rings (Invarianza Rotacional Matemática)
- **Qué se probó:** Arquitectura de anillos concéntricos (32 anillos, 64 muestras) donde cada anillo calcula la magnitud de la FFT (descartando fase) para invariancia a desplazamientos cíclicos.
- **Setup:** MNIST, 32 anillos x 64 muestras.
- **Resultado principal:** **62.59% idéntico en 0°, 90° y 180°** (invarianza perfecta).
- **Hallazgo:** Invarianza rotacional matemáticamente perfecta, pero la pérdida de fase reduce la precisión base a ~62% (no puede distinguir 6 de 9 si son rotaciones perfectas).

#### V119 — The Invariant Hybrid King
- **Qué se probó:** Arquitectura de 3 lóbulos: Invariante (Rings FFT), Estructural (Islas + Triangular) y Orientación (Mini-Raster Walsh) para distinguir 6/9.
- **Setup:** MNIST, 3,322 params.
- **Resultado principal:** **92.01%** a 0°; 89.34% a 15°; 45.13% a 90°.
- **Hallazgo:** Recupera >90% de precisión base. Robusto a pequeñas rotaciones (-2.7% a 15°). Más allá de 30°, los lóbulos de orientación/estructura dan información conflictiva que arrastra al lóbulo invariante.

#### V120 — Radical Cosine Experiment
- **Qué se probó:** Usar coseno/seno de la suma como activación en un MLP estándar (64 hidden).
- **Setup:** MNIST, MLP 64 hidden, 10 épocas.
- **Resultado principal:** ReLU: 97.28%; Pure Cosine: 96.45%; Pure Sine: 96.59%; Cosine+ReLU: 96.13%.
- **Hallazgo:** Las activaciones periódicas arrancan más rápido (93.98% en época 1 vs 93.37% ReLU) pero fluctúan más al final. ReLU(cos(z)) es la peor (pierde la fase negativa). Las periódicas son viables pero requieren tuning cuidadoso.

#### V121 — Projection Sinusoids
- **Qué se probó:** Arquitectura "Radon-inspired": pesos espaciales fijos a 1 (sumas de filas/columnas) + moduladores sinusoidales aprendibles (frecuencia y fase).
- **Setup:** MNIST, 28 filas + 28 columnas → 8 neuronas seno por proyección → Linear(448, 10); 5,386 params.
- **Resultado principal:** **88.87%**.
- **Hallazgo:** Las proyecciones 1D contienen ~90% de la información de clasificación. Cero pesos espaciales aprendidos. El techo de 88-89% se debe a la falta de correlación cruzada 2D (saber DÓNDE está el píxel en la fila).

#### V122 — Smooth Walsh Neurons
- **Qué se probó:** Parametrizar pesos en espacio Walsh de baja resolución y reconstruir con interpolación bilineal (filtro paso-bajo aprendible).
- **Setup:** MNIST, 10 épocas, K=8/16.
- **Resultado principal:** **Smooth Walsh K=16: 98.13%** (supera al Dense 97.88% con 4x menos params); K=8: 96.91% con 9,866 params.
- **Hallazgo:** El suavizado es esencial: el Walsh "blocky" puro sufre aliasing. Los pesos suaves parecen filtros Gabor orgánicos. Recomendación: usar Smooth Walsh/DCT como método de síntesis por defecto.

#### V123 — Fair Smooth Comparison (Walsh vs DCT)
- **Qué se probó:** Igualar params entre variantes para aislar el efecto del método de reconstrucción (smoothing vs blocky) y la base (Walsh vs DCT).
- **Setup:** MNIST, 10 épocas, 128 hidden, K=8/16.
- **Resultado principal:** Walsh Smooth K=8: 97.07% (+0.9% vs blocky); DCT Pure K=8: **97.56%**; DCT Smooth K=8: 97.01% (degrada).
- **Hallazgo:** **Walsh ama el smoothing** (interpolación bilineal esencial); **DCT ama la pureza** (padding espectral, la interpolación bilineal rompe sus propiedades trigonométricas).

#### V124 — Micro Walsh Neurons (K=2, K=4)
- **Qué se probó:** Límites absolutos de compresión con Walsh espectral a resoluciones ultra-bajas.
- **Setup:** MNIST, 10 épocas, K=2/4/8.
- **Resultado principal:** Blocky K=2: 52.64%; Smooth K=2: 30.93% (fallo); **Smooth K=4: 90.18%** (+7.7% vs blocky); Smooth K=8: 97.19%.
- **Hallazgo:** K=4 es la resolución mínima para visión "significativa" con neuronas suaves. En K=2, el suavizado produce un gradiente casi lineal que elimina demasiada estructura. K=8 es el mejor trade-off.

#### V125 — Smooth Spectral Adam (SWO)
- **Qué se probó:** Comprimir los estados de Adam (m y v) usando interpolación bilineal (proxy de Smooth Walsh).
- **Setup:** MNIST, 2 épocas, K=0.25/0.125.
- **Resultado principal:** **K=0.25: 95.94% con 93.6% menos RAM** (0.261 MB vs 4.088 MB); K=0.125: 90.45% con 98.3% menos.
- **Hallazgo:** ~94% de la información en los estados de Adam es ruido de alta frecuencia redundante. La reconstrucción suave del segundo momento actúa como regularizador implícito. Overhead de cómputo: solo +5%.

#### V126 — Total Spectral Entropy
- **Qué se probó:** Combinar arquitectura espectral (Smooth Walsh) + optimizador espectral (SWO): "entrenamiento total espectral".
- **Setup:** MNIST, 5 épocas, 512 hidden.
- **Resultado principal:** Params: 3.1x menos (168,714); **RAM optimizador: 51x menos (82.38 KB)**; Memoria total: 8.4x menos; Accuracy: 90.40%.
- **Hallazgo:** El "manifold de optimización" es significativamente más pequeño que el espacio de parámetros. Permite entrenar modelos complejos en microcontroladores/edge IoT.

#### V127 — Adaptive Recursive Spectral Optimizer (ARSO)
- **Qué se probó:** Escalado recursivo de la resolución del optimizador espectral (K=0.25 → 0.5) cuando la pérdida se estanca.
- **Setup:** MNIST, 8 épocas.
- **Resultado principal:** Paridad con V126 (89.75% vs 90.40%). Transición delicada: requiere halving del LR.
- **Hallazgo:** El salto de resolución sin interpolación+damping colapsa el modelo. La interpolación log-space es inestable (usar lineal). El mecanismo es "production-ready" pero el beneficio no justifica la complejidad en MNIST.

#### V128 — LLM Spectral Scanning (GPT-2)
- **Qué se probó:** Primer análisis espectral de un LLM real (GPT-2) para medir la redundancia de sus matrices de pesos en el dominio de frecuencias.
- **Setup:** GPT-2, transformadas Walsh/FFT/DCT sobre capas de atención y MLP.
- **Resultado principal:** **Walsh compacta el 50% de la energía con solo ~12% de coeficientes** (vs 18% FFT).
- **Hallazgo:** Los pesos de LLMs no se comportan como imágenes naturales (DCT); tienen estructura discreta de alta sequency que se alinea con Hadamard. La redundancia espectral es universal en profundidad.

#### V129 — LLM Spectral Pruning (GPT-2)
- **Qué se probó:** Podar coeficientes Walsh de GPT-2 y medir la coherencia lingüística.
- **Setup:** GPT-2, pruning al 50%/25%/10%.
- **Resultado principal:** **50% (2x): coherente** ("The capital of France is the city of Paris"); 25% (4x): repetitivo; 10%: ruido.
- **Hallazgo:** Umbral mágico del 50%: GPT-2 sobrevive a 2x compresión zero-shot sin fine-tuning. Transición de fase abrupta entre 50% y 25%. Los componentes de baja energía NO son ruido: llevan el "fine-tuning" lingüístico.

#### V130 — Block-Based Spectral Pruning (GPT-2)
- **Qué se probó:** Empujar el límite de pruning más allá del 50% usando transformadas Walsh por bloques locales (64x64 → 16x16) y rescaling de varianza.
- **Setup:** GPT-2, bloques 16x16/64x64, 25% de coeficientes.
- **Resultado principal:** A 25%: colapso de coherencia (aunque los errores son "más inteligentes" — palabras semánticas).
- **Hallazgo:** El rescaling de varianza es el ancla (sin él, salida vacía). La localidad mejora el "sabor" semántico pero no supera la pérdida de relaciones de fase. **Límite zero-shot estable: 2x**.

#### V131 — Spectral Quantization (GPT-2)
- **Qué se probó:** Cuantizar 1-bit y 2-bit los coeficientes Walsh de GPT-2 como alternativa al pruning.
- **Setup:** GPT-2, cuantización 1-bit/2-bit Walsh.
- **Resultado principal:** **Colapso total** (solo puntuación).
- **Hallazgo:** La cuantización 1-bit amplifica el ruido: los coeficientes "background" (casi cero) se elevan a la magnitud media, convirtiendo la matriz en una máscara de ruido de alta entropía. El rango dinámico es crítico. No se puede cuantizar espectralmente un modelo entrenado en dominio espacial.

#### V132 — Universal Approximation Benchmark
- **Qué se probó:** Comparar MLPs densos vs "Neuronas Polimórficas" (65 params) en aproximación de funciones matemáticas.
- **Setup:** 7 funciones (x², 1/x, prod, sin, cos, tan, sinc), CPU.
- **Resultado principal:** Poly generaliza mejor en x² (test 1.23 vs 2.24 del MLP 4.3k); **falla críticamente en prod (1.77)**.
- **Hallazgo:** El bias inductivo correcto (base cuadrática) es más potente que el ancho de red. Pero las neuronas polimórficas operan en dimensiones independientes: necesitan interacción cruzada.

#### V133 — Interaction Polymorphic Neurons
- **Qué se probó:** Añadir un canal de interacción explícito (PROD: proyecciones duales multiplicadas) a las neuronas polimórficas.
- **Setup:** 153-225 params, 7 funciones.
- **Resultado principal:** **prod: 0.000049 (1000x mejor que V132)**; iguala al MLP 4.3k; test MSE 0.23 vs 2.00 del MLP.
- **Hallazgo:** El multiplicador físico permite entender la operación fuera del rango de entrenamiento. La división sigue siendo inestable (asíntotas). `nan_to_num` y `clamp` críticos.

#### V134 — Spectral Cerebellum Polymorph
- **Qué se probó:** Integrar un banco de Walsh 1D (Cerebelo Espectral) dentro de la neurona polimórfica.
- **Setup:** 289-361 params, 7 funciones.
- **Resultado principal:** **1/x: 41x mejor** (0.107); sin: 68x mejor; sinc: 20x mejor; tan: 2x mejor.
- **Hallazgo:** El canal espectral actúa como "corrector de errores" de alta frecuencia. La combinación base analítica + correcciones Walsh resuelve asíntotas que a los MLPs les cuestan miles de params. El dial de atención se desplaza al canal SPECTRAL en sin/tan.

#### V135 — Cognitive Hierarchy (Fast vs Slow Thinking)
- **Qué se probó:** Arquitectura de 2 etapas: Pensamiento Rápido (polimórfico analítico) + Pensamiento Lento (reflexión espectral) regulado por un Surprise Gate.
- **Setup:** 382-442 params, 7 funciones.
- **Resultado principal:** prod: gate 0.4% (fácil, solo fast); tan: gate 72.5% (difícil, slow); 1/x: gate 61.2%.
- **Hallazgo:** "Sparsity of Thought": la red elige no pensar complejamente si no es necesario. La compuerta aprende a identificar regiones difíciles del espacio de entrada (metacognición funcional).

#### V136 — Escalabilidad y Saturación Espectral
- **Qué se probó:** Benchmark de Smooth Walsh vs Dense a dimensión 8192 en GPU DirectML.
- **Setup:** Dim 8192, GPU Radeon 780M.
- **Resultado principal:** **2.6x más rápido** (17.30ms vs 45.81ms); params 532,480 vs 8,396,800.
- **Hallazgo:** El "Muro de Adam" domina: 97% del tiempo del modelo denso es el optimizador. Al reducir params, se reducen proporcionalmente los estados de Adam. La síntesis cacheada hace el forward espectral despreciable.

#### V137 — Humillando al MLP en Terreno Imposible
- **Qué se probó:** Encontrar el límite físico donde el MLP denso colapsa y demostrar la invulnerabilidad espectral.
- **Setup:** Dimensiones 16,384/32,768/131,072, GPU DirectML.
- **Resultado principal:** MLP colapsa (OOM) a 32k; **Spectral ejecuta 131,072 dims en 96 MB a 89.2ms**.
- **Hallazgo:** Ruptura de la barrera de memoria: el modelo espectral opera en dimensiones que requerirían un clúster para un MLP denso, todo en GPU integrada. La FWHT (N log N) es masivamente más eficiente que mover matrices de GB.

#### V138 — Memoria Holográfica Espectral (131k ítems)
- **Qué se probó:** Memoria asociativa de contenido (CAM) masiva usando firmas Walsh.
- **Setup:** 131,072 recuerdos, 50% ruido blanco, GPU DirectML.
- **Resultado principal:** **100% precisión con 50% ruido**; búsqueda en 16.09ms; throughput 8,148 recuerdos/ms.
- **Hallazgo:** Las firmas Walsh son altamente ortogonales y resistentes a interferencia. Búsqueda sin índices: comparación holográfica en una sola operación matricial espectral. Nota (revisión): por Parseval, el ranking por producto escalar es idéntico en Walsh que en píxeles — la FWHT no cambia resultados de búsqueda por similitud.

#### V139 — Holographic MNIST (Zero-Shot)
- **Qué se probó:** Clasificar MNIST sin entrenamiento guardando las 60,000 muestras como recuerdos holográficos espectrales.
- **Setup:** 60,000 recuerdos, 0 épocas, latencia 0.36ms/imagen.
- **Resultado principal:** **92.42%**.
- **Hallazgo:** La memoria como inteligencia funciona en visión estática. **Nota crítica (revisión)**: 92.42% está POR DEBAJO del baseline trivial 1-NN L2 sobre píxeles crudos (~96.9%, LeCun 1998). La causa probable: producto escalar/coseno en lugar de L2 (dominado por cantidad de tinta). Es un resultado negativo, no un triunfo.

#### V140 — Holographic-PAC
- **Qué se probó:** Combinar memoria espectral con el algoritmo PAC de purificación por bifurcación.
- **Setup:** MNIST, 60,000 → 203 arquetipos.
- **Resultado principal:** **92.84% con 295.6x compresión** (supera a la memoria completa de 60k).
- **Hallazgo:** El promedio en espacio Walsh filtra ruido de alta frecuencia creando arquetipos más robustos. Nota (revisión): con baseline corregido a 96.9%, PAC con 203 arquetipos pierde 4 puntos a cambio de 295x compresión — resultado legítimo de condensación, pero no "supera a la memoria completa".

#### V141 — Spectral PAC-V2 (Taxonomía de la Confusión)
- **Qué se probó:** PAC-V2 con purificación por pares de confusión en dominio espectral.
- **Setup:** MNIST, 960 arquetipos finales.
- **Resultado principal:** **93.83%** (Top-1); compresión 62.5x; ~0.3ms/imagen.
- **Hallazgo:** Crea "especialistas" en distinguir casos difíciles (4 vs 9, 7 vs 1). El 0 es la clase con más variantes de confusión. El crecimiento de arquetipos se satura orgánicamente (+88 → +52). Herramienta de auditoría de datos masiva.

#### V142 — Refined PAC (El Colapso de la Escultura)
- **Qué se probó:** Refinar los arquetipos PAC con gradientes (Adam) para alcanzar SOTA.
- **Setup:** MNIST, 80 arquetipos.
- **Resultado principal:** **Degrada**: 89.68% (PAC puro) → 84.39% (refinado).
- **Hallazgo:** Los gradientes destruyen la coherencia espacial de los arquetipos (ruptura ontológica). En dominio Walsh, pequeños cambios en coeficientes alteran drásticamente la resonancia. **La inteligencia está en la Taxonomía, no en el ajuste de pesos.**

#### V143 — CSI Espectral (Auditoría de Datos)
- **Qué se probó:** Usar la memoria de 131k (60k muestras MNIST) para detectar errores de etiquetado mediante consenso holográfico.
- **Setup:** MNIST, 60k muestras, auditoría completa.
- **Resultado principal:** **1,423 anomalías (2.37%)**; detecta el error real 59915 (Oficial: 4, Consenso: 7) documentado en la literatura.
- **Hallazgo:** La memoria holográfica puede "sanar" los datos: identifica ambigüedades estructurales (7 vs 1, 8 vs 1) y errores reales con 100% de confianza. Potencial de limpieza de datasets.

#### V146 — Hybrid Memory (Walsh + Islas)
- **Qué se probó:** Fusionar la potencia espectral de Walsh (1024D) con la información topológica de Island Signatures (56D) para memoria asociativa 1-NN.
- **Setup:** MNIST, 1080D, zero-shot (sin backprop).
- **Resultado principal:** **97.42%** (+0.19% vs solo Walsh 97.23%).
- **Hallazgo:** Las islas capturan conectividad (morfología) que Walsh no puede: visión "binocular" espectral + estructural. Las islas son resistentes al grosor del trazo. Cero entrenamiento: éxito de la arquitectura sobre el cómputo bruto.

### Era Memoria Holográfica (v150–v170)

> **Tema:** evolución de la memoria holográfica desde fuerza bruta (120k slots) hacia **resonancia de cristales** (compresión 937x), **expertos espectrales (MoE)**, el **LLM espectral V8** (matrix-free, 2500x compresión) y el **auto-arquitecto** (neurogénesis).
> **Hito clave:** V150 logra 97.68% zero-shot; V165 (Spectral-MoE) logra 93.27% con 138x compresión; V163n demuestra 2500x compresión vs Transformer denso; V170 (Auto-Arquitecto) logra 96.08%.

#### V150 — Resonancia Dual (La Victoria de la Escala)
- **Qué se probó:** Guardar dos versiones de cada dato (orgánica y estandarizada) en 120,000 slots de memoria holográfica.
- **Setup:** MNIST, 120k slots (de 131,072), ~500 MB VRAM, zero-shot.
- **Resultado principal:** **97.68%** (vs 97.42% de V146).
- **Hallazgo:** Invarianza por redundancia: las dos vistas se complementan (una gana con imágenes deformadas, la otra con imágenes limpias). Uso extremadamente eficiente de la VRAM. Cero backprop.

#### V151–V156 — Élite PAC y Cerebro Fluido
- **Qué se probó:** Destilar los 120k recuerdos: V151-V152 (élite PAC con invarianza sintética) y V156 (Cerebro Fluido: arquetipos que mutan con EMA update).
- **Setup:** MNIST, 10,000 → 30,000 slots.
- **Resultado principal:** V151-V152: 93.23%; **V156: 97.03% con solo 30,000 slots**.
- **Hallazgo:** La memoria dinámica (EMA) es 2x más eficiente que la estática con pérdida despreciable. La invarianza sintética ayuda pero no sustituye la densidad de datos.

#### V157–V161 — La Revolución de los Cristales
- **Qué se probó:** Abandonar la búsqueda 1-NN por superposición holográfica: cristales 3D, multiplexación, clanes morfológicos, manifold ordenado (Greedy TSP) y atención Hopfield S^12.
- **Setup:** MNIST, de 10 objetos volumétricos a 1 bloque de 64 canales.
- **Resultado principal:** V157: 62.87%; V158: 62.99% (156x); V159: 77.15% (+14% con clanes); V160: 83.45% (937x); **V161: 83.97%** (techo).
- **Hallazgo:** La interferencia destructiva es el mayor enemigo (división en clanes obligatoria). Tratar la memoria como manifold ordenado permite que Walsh capture la dinámica de transformación.

#### V162 — Meta-Abstracción
- **Qué se probó:** Clasificar mediante el "Ritmo de Resonancia" (Meta-Walsh).
- **Setup:** MNIST.
- **Resultado principal:** **71.32%**.
- **Hallazgo:** Inferior para clasificación, pero demuestra que el perfil de resonancia de una imagen es una firma estructural única.

#### V163 — Spectral-FFN
- **Qué se probó:** Arquitectura de 3 pasos (Proyección → Activación Hopfield S^16 → Síntesis) para emular un FFN de Transformer.
- **Setup:** MNIST, 256 clanes, 262,144 floats (234x compresión).
- **Resultado principal:** **91.94%**.
- **Hallazgo:** La no-linealidad de alta potencia (S^16) es fundamental para "enfocar" la memoria cuando la densidad de conceptos aumenta.

#### V164 — Profundidad y Residuos
- **Qué se probó:** "Pensamiento Lento" con 2 capas jerárquicas donde la segunda analiza el residuo (error) de la primera.
- **Setup:** MNIST.
- **Resultado principal:** **91.99%** (+0.05%).
- **Hallazgo:** El residuo espectral puro es difícil de clasificar si la primera capa ya extrajo la estructura principal. La jerarquía necesita comunicación más rica que la resta de señales.

#### V165 — Spectral-MoE (HITO)
- **Qué se probó:** Mixture of Experts espectral: un Router global selecciona candidatos y delega a expertos de clase.
- **Setup:** MNIST, 458,752 floats (138x compresión).
- **Resultado principal:** **93.27%**.
- **Hallazgo:** La especialización morfológica es la clave para escalar: entrenar expertos solo en su clase elimina la interferencia destructiva entre números visualmente distintos.

#### V166 — Auto-Crítica (Análisis por Síntesis)
- **Qué se probó:** El sistema reconstruye su "imagen mental" del número y la compara con la realidad.
- **Setup:** MNIST, 32 clanes por experto.
- **Resultado principal:** **91.58%**.
- **Hallazgo:** Un sistema que se autocorrige necesita una base de conocimientos de altísima fidelidad. Con 32 clanes, el "sueño" es demasiado impreciso para ser juez fiable.

#### V163b–V163e — La Evolución hacia el LLM Espectral
- **Qué se probó:** Escalado masivo: V163b (4,096 expertos), V163c (multiplexado por suma), V163d (MoE extremo 131,072 expertos), V163e (multiplexado espaciotemporal con Roll).
- **Setup:** MNIST/CPU, 131,072 expertos, 517 MB RAM.
- **Resultado principal:** **V163b: 95.68%**; V163c: colapso (SNR < 1.5); V163d: 308 tok/s CPU; V163e: recuperación perfecta de 4 tokens.
- **Hallazgo:** La inteligencia espectral es función del ancho del manifold. No se puede "apilar" información sin clave (interferencia destructiva). El orden es una dimensión espectral (Roll crea ortogonalidad temporal).

#### V163f — Stress Test de Memoria Holográfica
- **Qué se probó:** Punto de ruptura de la memoria holográfica con Roll sin atención.
- **Setup:** D=512-2048, contexto L=128-512.
- **Resultado principal:** Colapso rápido: 5% (D=512, L=128) → 0% (D=2048, L=512).
- **Hallazgo:** Sin filtrado, la memoria se satura extremadamente rápido. Doblar la dimensión ayuda pero no resuelve el ruido acumulado. La saliencia es obligatoria.

#### V163g — Holographic Attention (El Poder de la Saliencia)
- **Qué se probó:** Aplicar un peso de atención W al token "aguja" y peso 1.0 al ruido.
- **Setup:** Contexto L=1024-4096, D=1024-2048.
- **Resultado principal:** **100% recall a 4,096 tokens con W=20** (SNR 9.62); W=100: SNR 27.07.
- **Hallazgo:** La atención es el filtro: un peso de 20x permite que un token sobreviva a 4,000 tokens de interferencia. Ruptura del límite cuadrático: memoria constante con contexto masivo.

#### V163h — Sentence Recall (Compresión de Frases)
- **Qué se probó:** Recuperar una frase de 8 tokens (peso 30x) rodeada de 2,000 tokens de ruido (peso 1x).
- **Setup:** D=2048, frase de 8 tokens, 2,000 tokens de ruido.
- **Resultado principal:** **100% fidelidad**; recuperación posicional perfecta con Roll.
- **Hallazgo:** El desplazamiento circular (Roll) preserva el orden de forma extremadamente robusta (no confunde "A luego B" con "B luego A"). Una idea secuencial compleja comprimida en un solo vector.

#### V163j–V163n — Spectral V8.3–V8.6 (Matrix-Free)
- **Qué se probó:** Optimización del motor espectral: V8.3 (sweep 60 configs), V8.4 (Spectral Residency: mantener estado en dominio Walsh), V8.5 (motor C++ nativo OpenMP+AVX2), V8.6 (Universal Core CPU/GPU).
- **Setup:** Dim 32,768, 16-24 capas, 128-256 expertos, Ryzen 7 8845HS + Radeon 780M.
- **Resultado principal:** V8.3: 1.42 tok/s; V8.4: 4.92 tok/s (3.46x); V8.5: 4.57 tok/s (C++); **V8.6: 10.63 tok/s GPU (batch 16)**.
- **Hallazgo:** Residencia espectral elimina 98% de las FWHT. **Compresión ~2500x vs Transformer denso** (D=32768: 21.3B params densos vs 8.4M espectrales). CPU mejor para latencia (batch 1); GPU para throughput (batch 16+).

#### V167–V170 — La Era del Auto-Arquitecto
- **Qué se probó:** Neurogénesis residual: la red se "cultiva" añadiendo capas de especialistas solo donde hay errores recurrentes (inyección de capas identidad).
- **Setup:** MNIST, 2-4 capas, 512-1024 clanes.
- **Resultado principal:** V167: 93.85%; V168: 91.42%; V169: 95.26%; **V170: 96.08%**.
- **Hallazgo:** Inyección orgánica sin olvido catastrófico (las capas nuevas nacen para corregir a las anteriores). Solo usa parámetros para lo difícil. Escalabilidad infinita sin desvanecimiento de gradiente (sistema aditivo residual).

#### V168 — Vectorización Holográfica y MoE Jerárquico
- **Qué se probó:** Transición a V8.1.2: Flash-Hologram (vectorización con gather/cumsum) + MoE jerárquico por clanes (512 clanes x 256 especialistas).
- **Setup:** Radeon 780M iGPU, DirectML, 230M params.
- **Resultado principal:** **6.8x speedup** (71.9s vs 494s por iter); loss 5.15 tras eliminar weight_decay; capacidad equivalente a 1.1T params densos.
- **Hallazgo:** La vectorización elimina el overhead de lanzamiento de kernels (12 min → 72s). El gating jerárquico por clanes reduce la presión de VRAM. **El weight_decay es un "freno de mano"** en arquitecturas factorizadas (la compresión estructural ya regulariza).

### Era Resonancia (v171–v199)

> **Tema:** sustitución de capas densas por **escaneos de resonancia armónica** y votación física, y la consolidación de las **neuronas polimórficas** (estructural + log + resonancia) para aproximación de leyes matemáticas con generalización OOD superior.
> **Hito clave:** V186 (Pure Resonance) logra 96.12% MNIST con 75k params; V188 (Voting Resonance) logra 87.34% con solo 1.1k params; V192 logra 32,000x mejor estabilidad en Schwefel; V195 resuelve el módulo con 400x menos params.

#### V171–V189 — La Era de la Resonancia y el Voto Armónico
- **Qué se probó:** Sustituir capas densas por osciladores armónicos (Trig Symphony, Total DCT, Pure Resonance, Voting Resonance) y sistemas de votación física.
- **Setup:** MNIST, de 1.1k a 203k params.
- **Resultado principal:** V171: 93.80% (203k); V177: 82.93% (4.7k); **V186: 96.12% (75k)**; **V188: 87.34% (1.1k)**.
- **Hallazgo:** La inicialización determinista (ordenada) derrota al caos (V186: del 11% al 96%). Se puede clasificar sin capas densas, solo por sintonía (V188). **Independencia de resolución**: de 28x28 a 800x800 sin añadir params (solo muestrear más puntos).

#### V190 — Structural Generalization Benchmark
- **Qué se probó:** Benchmark OOD de funciones matemáticas (x², sinc, prod, Schwefel) comparando MLPs vs Neurona Polimórfica.
- **Setup:** 161-465 params (Poly) vs 4,353-132,865 (MLP).
- **Resultado principal:** Poly compite o supera en extrapolación: Schwefel ratio 2.16 vs 3,431 del MLP-L.
- **Hallazgo:** Los MLPs hacen interpolación estadística (error OOD explota a ratios de 10⁷); la Poly "entiende" la ley subyacente. Compresión de conocimiento de 300x-800x.

#### V191 — Log-Polymorphic Interaction
- **Qué se probó:** Rama logarítmica (log-linear-exp) para linealizar productos y divisiones.
- **Setup:** 16 neuronas, funciones div/prod/gravity.
- **Resultado principal:** **64x más estable en div** (ratio 1,690 vs 109,000 del MLP); 4x mejor en prod.
- **Hallazgo:** La linealización logarítmica permite extrapolar x/y con coherencia. La rama de signos (tanh) ayuda pero el producto de signos es una operación XOR discreta difícil para redes continuas.

#### V192 — Resonant-Log-Polymorphic Integration
- **Qué se probó:** Unificar las tres ramas: estructural (polinómica) + logarítmica + resonancia (para periodicidad).
- **Setup:** 881-1,425 params, funciones Schwefel/sin(5x)/Rastrigin/Ackley.
- **Resultado principal:** **Schwefel: ratio 25.1 (32,000x mejor que MLP-L)**; sin(5x): 38x mejor; Rastrigin: 2.5x; Ackley: 2x.
- **Hallazgo:** La resonancia es el antídoto al olvido OOD en funciones periódicas. La sinergia de las tres ramas captura la ley de formación.

#### V193 — Deep Polymorphism
- **Qué se probó:** Profundidad multicapa en la arquitectura polimórfica.
- **Setup:** 2 capas, 5,521 params, Rastrigin/Schwefel.
- **Resultado principal:** Rastrigin: 0.05 (2x mejor que flat); **Schwefel: ratio 0.998** (error independiente del rango).
- **Hallazgo:** El polimorfismo es componible (las capas trabajan en secuencia). Un ratio de 0.998 es el nivel máximo de "comprensión" algorítmica. Desafío: optimización más difícil (vanished gradient).

#### V194 — Modulus Challenge (La Pared de la Discontinuidad)
- **Qué se probó:** Función módulo (x % y), una de las más difíciles por discontinuidades abruptas.
- **Setup:** MLP-Huge (1,052,673 params) vs Poly-Deep (28,385).
- **Resultado principal:** MLP-Huge: 0.0055 local (12x mejor); **Poly: 3x más estable OOD** (ratio 229 vs 743).
- **Hallazgo:** La fuerza bruta gana en rango corto (memorización con millones de ReLU), pero la Poly intenta capturar la periodicidad. El floor (⌊x/y⌋) es el talón de Aquiles de las arquitecturas continuas.

#### V195 — Discontinuity Branch (Rompiendo el Muro del Módulo)
- **Qué se probó:** Rama sawtooth nativa con Straight-Through Estimator (STE) para gradientes a través de saltos.
- **Setup:** 2,513 params, función módulo.
- **Resultado principal:** **0.0095 train MSE (similar a MLP-Huge) con 400x menos params**.
- **Hallazgo:** El STE permite optimizar frecuencias/fases de "dientes de sierra" a pesar del gradiente cero. Desafío: la extrapolación en saltos sufre desincronización de fase (pequeño error de frecuencia → error masivo OOD).

#### V196 — The Recursive Compression Paradox
- **Qué se probó:** Comprimir los coeficientes de una compresión previa (Walsh → DCT).
- **Setup:** Señal 1D, N=64.
- **Resultado principal:** **Fallo masivo**: MSE 303.51 (vs 0.1026 directo).
- **Hallazgo:** Las transformadas espectrales funcionan porque los datos tienen correlación espacial. Tras transformar, los coeficientes están decorrelacionados; comprimir ruido blanco dispersa la energía. La excepción son las wavelets (comprimir solo la rama de baja frecuencia).

#### V197 — Lateral Interaction (Dinámica Padre-Hijo)
- **Qué se probó:** Neuronas de la misma capa se combinan con operaciones simbólicas (+, -, ·, mod) para crear "neuronas hijas".
- **Setup:** 967 params, función (x*y) % (x+y).
- **Resultado principal:** Loss final **1.65** (desde 4.39); gating sintonizado hacia Suma/Módulo.
- **Hallazgo:** Jerarquía funcional dentro de una sola capa (razonamiento simbólico instantáneo). El Softmax gating promedia operaciones al inicio (lento); Gumbel-Softmax forzaría elección discreta.

#### V198 — Entropy-Spectral Hybrid (La Victoria Sin Pérdida)
- **Qué se probó:** Segunda compresión sin pérdida (Huffman) sobre coeficientes espectrales cuantizados.
- **Setup:** N=256, Top-K=32, 8-bit + Huffman.
- **Resultado principal:** **21.28x compresión total** (8,192 → 385 bits); MSE 5.73e-02. A escala GB: 85x (1TB → 11.76GB).
- **Hallazgo:** Huffman funciona porque el Top-K genera distribución sesgada (muchos ceros). Es la base de Deep Compression: Pruning + Quantization + Huffman.

#### V199 — Competence-Based MoE (El Oráculo de Error)
- **Qué se probó:** 2 redes paralelas + un gater (minired) que predice cuál tendrá mejor loss.
- **Setup:** Función híbrida seno/parábola, 2 expertos + gater.
- **Resultado principal:** MoE: **0.186** (mejor que expertos individuales 0.186/0.190); gater asigna 0.65 al Experto A en x<0, 0.60 al B en x>0.
- **Hallazgo:** El gater actúa como predictor de error, aprendiendo las "zonas de confort" de cada experto. Ideal para funciones que cambian de régimen. Coste de elegir experto despreciable.

### Era Fase/Periódica (v200–v227)

> **Tema:** neuronas de **resonancia de fase** (interferencia de ondas), el "Firewall Biológico" contra ruido, la batalla por resolver el módulo (discontinuidades), el **MoE Honesto** (competición darwiniana entre espacios lineal/logarítmico) y la **conciencia artificial** (detección de novedad).
> **Hito clave:** V202 resuelve XOR con 41 params; V203 logra 96.22% MNIST en 5 épocas; V212 logra el récord de estabilidad OOD (ratio 8.61); V221 logra 100% de precisión filtrada con abstención.

#### V200 — Bounded Parameters (Unidades Lego para 8-bit)
- **Qué se probó:** Restringir todos los pesos y bias al rango [-1, 1] mediante tanh, con un factor de escala aprendible por neurona.
- **Setup:** Red polimórfica, función objetivo.
- **Resultado principal:** Loss **1.42e-05**; pesos en [-0.995, 0.998].
- **Hallazgo:** Desacoplamiento de magnitud y dirección: el núcleo acotado es mapeable a 8-bit puro; solo el factor de escala (1 por neurona) requiere precisión. Ahorro de memoria ~75% vs Float32.

#### V202 — Resonancia de Fase (Solución del XOR)
- **Qué se probó:** Modificar la neurona de resonancia: activación ReLU (filtro de interferencias constructivas) + Sigmoid/BCE + 8 características.
- **Setup:** XOR, 41 params.
- **Resultado principal:** **XOR resuelto con accuracy 100%** (loss 1.42e-05).
- **Hallazgo:** La neurona de resonancia basada en interferencia de fase (coseno) es Turing-completa en la práctica para lógica no lineal con un umbral asimétrico (ReLU). El cerebro puede operar por sintonización de frecuencias.

#### V203 — Escalado de Resonancia a MNIST
- **Qué se probó:** FastResonantLayer usando la identidad trigonométrica cos(x-w) = cos(x)cos(w) + sin(x)sin(w) para evitar el tensor 3D (OOM).
- **Setup:** MNIST, 784→128→10, 203,520 params, 5 épocas, Adam lr=0.005.
- **Resultado principal:** **96.22%** (76.61% → 95.90% → 96.22%).
- **Hallazgo:** Mapear intensidad de píxel a fase (ángulo) + resonancia armónica constructiva proporciona gradientes estables y generalización. La identidad trigonométrica hace el cálculo OOM-free.

#### V204 & V205 — El Firewall Biológico y el Phase Jitter
- **Qué se probó:** Robustez al ruido de las neuronas resonantes. V204 falló (colapso con ruido std=0.5); V205 corrigió con escala angular π/4 y vacunación por ruido.
- **Setup:** MNIST, ruido gaussiano std 0-1.0.
- **Resultado principal:** V205: **87.72%** (ruido 0), 87.59% (0.5), **74.88%** (1.0). V204 colapsó a 10-11%.
- **Hallazgo:** El "Phase Jitter" es real: mapear intensidad a π convierte ruido aditivo en desfase de 180° (inversión de señal). Escalar a π/4 y entrenar bajo estrés crea un verdadero firewall.

#### V206 — Resonancia Espectral (Compresión DCT + Fase)
- **Qué se probó:** Proyectar la imagen con DCT-II 2D fija (K=64 frecuencias bajas) antes de la capa resonante.
- **Setup:** MNIST, 19,200 params (vs 203,500 originales, >10x compresión), ruido std=0.5.
- **Resultado principal:** **91.96%** (época 4); robustez: 89.38% (ruido 0.5), 74.87% (1.0).
- **Hallazgo:** La filtración DCT actúa como "nervio óptico" limpiando ruido de alta frecuencia. El firewall biológico se mantiene: ruido 0.5 apenas afecta (-2.6%).

#### V207 — El Desafío del Módulo con Resonancia de Fase
- **Qué se probó:** Función módulo (x % y) con neuronas de fase (cosenos).
- **Setup:** 17,089 params.
- **Resultado principal:** Train MSE 0.0612 (mejor que Poly-Deep); **OOD MSE 29.62** (ratio 484).
- **Hallazgo:** Las ondas son excelentes para la forma de sierra local (serie de Fourier), pero la frecuencia del módulo depende de y (frecuencia ∝ 1/y). Es matemáticamente imposible que w₁x + w₂y represente x/y para todos los valores. Las neuronas necesitan modular dinámicamente sus frecuencias.

#### V208 — Explosión Multiplicativa y el Fenómeno de Gibbs
- **Qué se probó:** Modulación dinámica de frecuencia con exp(W·log(x)) para descubrir x/y.
- **Setup:** 753 params.
- **Resultado principal:** Train MSE 0.0551 (mejor); **OOD MSE 551.76 (colapso catastrófico)**.
- **Hallazgo:** El gradiente aprende aproximaciones sucias (x^0.95 · y^-1.02) que divergen exponencialmente en OOD. El Fenómeno de Gibbs: ajustar un salto con cosenos genera oscilaciones violentas. Combinar log/exp con ondas para discontinuidades es "receta para el desastre".

#### V209 — Neuronas Sawtooth y el Límite Lineal
- **Qué se probó:** Sustituir el oscilador cos por un oscilador discontinuo nativo (Sawtooth = p - round(p)).
- **Setup:** 17,473 params.
- **Resultado principal:** Train MSE 1.07 (no ajusta local); **OOD MSE 30.48 (ratio 28.4, el mejor de resonancia)**.
- **Hallazgo:** La función sawtooth es estrictamente lineal a trozos (sin curvatura). No puede aproximar la no-linealidad de x/y localmente, pero al no sobreajustar, tiene la mejor estabilidad relativa. Rigidez matemática = estabilidad.

#### V210 — La Neurona Analítica Discreta (El Rey del OOD)
- **Qué se probó:** Pesos estrictamente enteros (STE) para fase discreta + magnitud continua + sawtooth.
- **Setup:** 7,521 params.
- **Resultado principal:** **Mejor Far OOD MSE (13.15)** de todas las redes eficientes; **ratio 19.5** (vs 229 de Poly-Deep).
- **Hallazgo:** Fase Discreta + Magnitud Continua + Sawtooth es la fórmula maestra para extrapolación pura. El precio: Train MSE 0.67 (superficie de pérdida escalonada difícil para Adam). Se necesitaría optimización simbólica/evolutiva.

#### V212 — Optimización Simbólica con DGE
- **Qué se probó:** TorchDGEOptimizer (libre de gradientes) a través de operadores no-diferenciables (+, *, %, floor).
- **Setup:** 177 params, función módulo.
- **Resultado principal:** **Ratio 8.61 (récord absoluto de estabilidad)**; Train MSE 17.74.
- **Hallazgo:** El secreto del DGE es mantener parámetros continuos y dejar que un δ minúsculo desplace las activaciones a través de las barreras de los operadores. Cuanto más dura/lógica es la arquitectura, menos degenera en el infinito.

#### V213 — El "Activation Bridge" y la Trampa de Sísifo
- **Qué se probó:** Inyectar gradientes sintéticos (diferencias finitas) a un bloque simbólico no-diferenciable (%) para que Adam lo entrene.
- **Resultado principal:** **Fallo espectacular**: error oscilando entre 1.2 y 1300 (final 387.98).
- **Hallazgo:** El puente funcionó perfectamente; el problema es el paisaje de pérdida. El módulo genera una onda de sierra infinita: Adam empuja la piedra a la cima y la inercia la hace caer por el acantilado. **Los optimizadores locales no pueden con paisajes periódicos/discontinuos.**

#### V214 — El Colapso de los Expertos (Expert Collusion)
- **Qué se probó:** MoE con Softmax suave entre experto Lineal y Logarítmico.
- **Resultado principal:** Resta → 96.8% al Logarítmico (¡que solo produce positivos!); Train MSE 0.37; OOD 7800 (explosión).
- **Hallazgo:** Con Softmax suave, los expertos no compiten: **colaboran para anular sus errores** (el Log escupe +10, el Lineal aprende -13). En OOD, el exp() explota y el Lineal no puede compensar. Se necesita Hard-Routing (Top-1/Gumbel).

#### V214–V217 — La Batalla de los Espacios (Lineal vs Logarítmico)
- **Qué se probó:** Evolución del MoE: V214 (Soft, colapso) → V215-V216 (Hard/Warmup) → V217 (Honesto: competición darwiniana por error mínimo).
- **Resultado principal:** Mapa de verdad: Multiplicación→LOGARÍTMICO; Resta→LINEAL; Suma→LOGARÍTMICO (sorpresa); División→LINEAL (paradoja).
- **Hallazgo:** La representación (Input Mapping) es más poderosa que el número de capas. La mejor forma de forzar especialización es la competición darwiniana (solo el que tiene menos error recibe gradiente). La división evita log/exp por el "suelo de ruido" de las funciones trascendentales.

#### V218 — Compositional Attention Neuron (CAN)
- **Qué se probó:** 2 capas con MoE honesto (Mapper: Linear vs Log; Operator: MLP vs Harmonic) para sin(x·y).
- **Resultado principal:** **OOD MSE 0.87** (vs 2.89 del camino Lin→MLP, 333% mejor); PEI 0.4684; dominancia Log→Har 70.3%.
- **Hallazgo:** La red descubrió autónomamente que Log→Har es el camino óptimo (la multiplicación se convierte en suma en log-space). La generalización OOD se consigue con alineación estructural (Architecture-Problem Fit), no con más datos.

#### V219 — Conscious Attention Neuron (Auto-Confianza)
- **Qué se probó:** Doble cabeza (valor MSE + confianza log-MSE) con desacoplamiento de gradientes.
- **Resultado principal:** Train: MSE 6.13e-04, confianza 1.76e-04 (correlación 0.25); OOD: MSE 1.31, confianza 1.59e-05 (**colapso de confianza**).
- **Hallazgo:** El "Efecto Arrogancia" (Dunning-Kruger neural): la red predice un error MENOR cuanto más se equivoca. La confianza predictiva entrenada no sirve para OOD; debe ser una propiedad emergente o estructural (memoria/novedad).

#### V220 — Familiarity Atlas (Detección de Novedad)
- **Qué se probó:** Memoria de prototipos espectrales (centroides DCT 12x12 de las 10 clases) + Delta Encoding + métrica de familiaridad F = exp(-dist/scale).
- **Resultado principal:** Distancia media normal 4.46 vs rotado 5.17; **ratio de discriminación 7.5x**.
- **Hallazgo:** La familiaridad es una propiedad geométrica, no una predicción. El Delta Encoding (diferencias entre píxeles) hace la firma espectral mucho más sensible a la rotación. La red puede "saber que no sabe" sin entrenamiento OOD explícito.

#### V221 — Safe Attention Classifier (Muro de Seguridad)
- **Qué se probó:** Integrar el Atlas en la inferencia: si dist > 4.8, abstenerse.
- **Resultado principal:** **100% de precisión filtrada** (33% de abstinencia en datos limpios); 70.5% de rechazo en rotado 90°.
- **Hallazgo:** La Era de la Precisión Perfecta: si permitimos dudar basándonos en estructura, eliminamos los errores por ambigüedad. El agujero: el ruido puro no es rechazado (la distancia euclídea simple no detecta "ausencia de estructura").

#### V222 — Spectral Diffusion (MNIST)
- **Qué se probó:** Difusión generativa en el dominio DCT (SpectralDenoiseNet con MoE Lineal/Harmónico).
- **Setup:** 632,930 params, 200 pasos de tiempo, 10 épocas.
- **Resultado principal:** Loss 0.5896; muestras coherentes pero ruidosas.
- **Hallazgo:** La difusión en dominio DCT funciona y es estable. El modelo tiene acceso directo a la estructura global en cada paso (sin convoluciones). 10 épocas son insuficientes; se requiere escalado de coeficientes (σ=0.1).

#### V224 — Benchmark de Neurona Periódica vs ReLU MLP
- **Qué se probó:** Activación σ(tan(x)) con 4 params vs MLP ReLU de 2,241 params para x mod 1.
- **Resultado principal:** MLP: 0.0144 (PEI 0.2708); Periódica: 0.0721 (**PEI 1.3275, 4.9x superior**).
- **Hallazgo:** La neurona periódica entiende la estructura cíclica nativamente. El error se debe a la forma sigmoidal vs rampa lineal, no a falta de entendimiento. Eficiencia extrema para sistemas ultra-comprimidos.

#### V225 — Enderezamiento de la Rampa Periódica
- **Qué se probó:** Corrección polinómica de tercer grado sobre la activación periódica (8 params totales).
- **Resultado principal:** **0.019 MSE con 8 params** (vs 0.014 del MLP de 2,241 params); PEI 1.0272.
- **Hallazgo:** Paridad de precisión con 99.6% menos parámetros. El polinomio de tercer grado contrarresta la curvatura del sigmoide, creando una rampa casi perfectamente lineal sin perder periodicidad.

#### V227 — Clasificación MNIST Espectral-Periódica
- **Qué se probó:** DCT-2D fijo (8×8, 64 coeficientes) + capa StraightPeriodic (64×8) + clasificador lineal.
- **Setup:** 1,034 params, 10 épocas, Adam lr=0.01.
- **Resultado principal:** **85.83%** (vs 44.80% de V226 con ruido); PEI 0.2854.
- **Hallazgo:** Las neuronas periódicas resuenan naturalmente con los coeficientes DCT. Sintonizar fase/frecuencia de cada armónico identifica patrones estructurales (bucles, líneas) con una fracción del coste de una CNN. Hito de eficiencia: ~86% con 1k params.

### Era Descubrimiento Matemático (v229–v257)

> **Tema:** compresión espectral de LLMs (GPT-2), optimizadores con estabilidad direccional (Adam-DS, Lion-DS, Muon, Sign-DS), descubrimiento de leyes matemáticas (Augmented Features + poda), y **redes congeladas con gating** (ternary weights, oligarquía).
> **Hito clave:** V248 logra descubrimiento perfecto de leyes (10^-12 a 10^-19); V244 (Muon) logra 99.60% MNIST; V251e logra 93.89% con solo 4,106 params; V256 (Ternary CNN) logra 85.07% con 394 params.

#### V229 — El Triunfo de la Cuantización Espectral
- **Qué se probó:** Cuantización espectral jerárquica (Walsh con orden de sequencialidad, 8/4-bit) vs cuantización espacial RTN en GPT-2.
- **Setup:** GPT-2 Layer 6 MLP.
- **Resultado principal:** **-4% MSE global, -3.5% MSE outliers, +62.2% preservación de outliers, -0.22 PPL**.
- **Hallazgo:** El dominio espectral redistribuye la energía de los outliers, generando "ruido holográfico" distribuido que el modelo tolera mejor que el truncamiento localizado espacial. El orden de sequencialidad es clave para proteger las bajas frecuencias.

#### V235 — La Curva de Elasticidad Espectral
- **Qué se probó:** Barrido de poda Top-K (magnitud) de coeficientes Walsh en GPT-2.
- **Setup:** GPT-2, ratios de 0.4 a 1.0.
- **Resultado principal:** **40% ahorro con +5.0 PPL** (ratio 0.60, punto óptimo); 50% → +49 PPL; 60% → colapso.
- **Hallazgo:** Anomalía en ratio 0.60 (más preciso que 0.70): efecto de "limpieza espectral" al eliminar ruido de alta magnitud pero frecuencia irrelevante. La información crítica está distribuida en el espectro (Top-K es la única forma de capturarla).

#### V236 — Robustez de la Compresión Espectral en GPT-2
- **Qué se probó:** Validar la poda Top-K en un dataset diverso (Ciencia, IA, Historia).
- **Setup:** GPT-2, ratios de 0.5 a 1.0.
- **Resultado principal:** **30% ahorro estable (+1.31 PPL)**; 40% → +12.69 (límite); 50% → colapso.
- **Hallazgo:** El 30% de la información en Walsh es redundante/ruido. Al pasar del 30% al 40%, la perplejidad se triplica: en ese 10% residen conexiones transversales críticas para coherencia multidominio. Recomendación: ratio 0.70 (324MB → 226MB).

#### V239 — Firmas Espectrales (El Triunfo de la Interferencia)
- **Qué se probó:** Exponer los k=8 componentes espectrales individuales de una neurona a la siguiente capa (en vez de la suma).
- **Setup:** MNIST, 1,354 vs 7,178 params.
- **Resultado principal:** **Suma: 78.50%**; Signature: 71.49%; Signature sin ReLU: 64.39%.
- **Hallazgo:** Separar las frecuencias ("un-baking the cake") es destructivo. La neurona espectral aprende patrones espaciales (arquetipos) de la interferencia constructiva/destructiva. La suma entrega una **Gestalt** (un todo mayor que la suma de sus partes). La compresión es optimización de señal, no solo de memoria.

#### V240 — El Triunfo de la Diferenciabilidad Mixta
- **Qué se probó:** MoE con Adam (experto analítico) + DGE (experto simbólico con %).
- **Setup:** Modulus Challenge, 0.13% de params con DGE.
- **Resultado principal:** **Train MSE 5.98e-02**; Far OOD 10.34; PEI 4.57; alta estabilidad.
- **Hallazgo:** Supera la "Trampa de Sísifo" (V213): Adam da la base continua, DGE optimiza los 6 params simbólicos. No necesitamos que toda la red sea derivable; solo las partes lógicas necesitan el optimizador adecuado.

#### V242 — Adam-DS (Directional Stability)
- **Qué se probó:** Modular el LR con la consistencia de signo temporal (DS-EMA).
- **Setup:** MNIST, 4 épocas.
- **Resultado principal:** **98.94%** (+0.14% vs Adam); loss -10.6%; overhead +10%.
- **Hallazgo:** El modelo "corre" en laderas consistentes y "camina con cuidado" en valles ruidosos. La métrica exp_avg_sign actúa como filtro de confianza. Estado de estabilidad en Int8 (solo +12.5% memoria).

#### V243 — Lion-DS (The Memory Master)
- **Qué se probó:** Usar el signo del momentum (estilo Lion) eliminando la varianza.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **99.38%** con 5 bytes/p (37.5% menos RAM que Adam).
- **Hallazgo:** Al usar sign(momentum), cada actualización tiene la misma magnitud, eliminando la necesidad de v. La normalización implícita: no necesita saber cuán grande es el gradiente, solo hacia dónde apunta con más frecuencia.

#### V244 — La Frontera Muon
- **Qué se probó:** Ortogonalización de la actualización (Muon) vs Adam/Lion.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **Muon: 99.60%** (récord); Lion-DS: 99.38%; Adam-DS: 99.39%.
- **Hallazgo:** La ortogonalidad es la clave para capas lineales 2D. El híbrido Lion-Muon-DS (98.7%) falló: la modulación de LR por parámetro destruye las propiedades espectrales de la matriz ortogonal. Recomendación: Muon para capas densas, Lion-DS para capas 1D.

#### V244 — Sign-DS (The Memory Ghost)
- **Qué se probó:** Eliminar el momentum, usar solo el signo del gradiente + DS-EMA.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **98.18%** con 2 bytes/p (75% menos RAM que Adam).
- **Hallazgo:** El Sign-SGD puro converge sin momentum (DS-EMA amortigua el ruido), pero pierde ~1.1% de accuracy. Opción extrema para RAM crítica (microcontroladores). Lion-DS sigue siendo el mejor balance.

#### V244 — El Duelo Final de Optimizadores
- **Qué se probó:** Comparativa exhaustiva: SGD, Sign-DS, Lion, Muon, SGD+Momentum, RMSprop, Lion-DS, Adam.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **SGD+Momentum: 99.69%** (mejor balance); Sign-DS: 98.24% (2b/p); Adam: 99.45% (8b/p).
- **Hallazgo:** El momentum lineal es superior a las normalizaciones por signo/varianza en redes de este tamaño. Sign-DS supera al SGD puro por ~3% con solo 2 bytes extra. Si la memoria es escasa, Sign-DS es el nuevo estándar.

#### V245 — Noise Robustness (Spectral vs MLP)
- **Qué se probó:** Robustez al ruido de labels (0-60%) de arquitectura espectral matrix-free vs MLP.
- **Setup:** MNIST, 5 épocas, ruido simétrico 0-60%.
- **Resultado principal:** Espectral: 96.13% → 92.66% (60% ruido); MLP: 97.45% → 94.15%. **PEI espectral >21 vs ~19 del MLP**.
- **Hallazgo:** Ambas arquitecturas son sorprendentemente resistentes (aprenden señal antes que ruido). El modelo espectral logra robustez similar con 6x menos params. El filtrado de baja frecuencia hace "denoising" implícito.

#### V246 — Augmented Feature Universal Approximator
- **Qué se probó:** Una neurona lineal procesa un conjunto rico de transformaciones no-lineales del input (expansión de base).
- **Setup:** 18 params, rango [-2,2] → test [-4,4].
- **Resultado principal:** **x³ descubierto** (6.34 vs 158.46 del MLP); x²: MLP gana (redundancia de bases).
- **Hallazgo:** La transparencia: la neurona "confiesa" qué ley ha encontrado (x0*x1: 1.0019 para multiplicación). El aumento de base + neurona lineal es un descubridor de leyes interpretable.

#### V248 — Pruned Augmented Feature Approximator
- **Qué se probó:** Aumento de base + regularización L1 + poda agresiva (umbral 0.05).
- **Setup:** 18 params, rango [-10,10] → test [-40,40].
- **Resultado principal:** **Descubrimiento perfecto**: x² (10^-12), x³ (10^-11), prod (10^-14), sin (10^-15). Mejoras de 10^16 a 10^19 vs MLP.
- **Hallazgo:** La poda agresiva elimina el ruido residual, dejando fórmulas puras (x0*x1: 1.0000, 1 base activa). El descubrimiento simbólico es exacto.

#### V249 — Deep Scientific Network
- **Qué se probó:** Arquitectura multicapa para descubrir composiciones de leyes (g(f(x))).
- **Setup:** Funciones que requieren 2 pasos lógicos.
- **Resultado principal:** **Gaussiana (e^-0.169x²) y Sin-Square (sin(0.172x²)) reconstruidas**; Quad+Sin separado en términos.
- **Hallazgo:** El paradigma "Aumento + Poda" es componible (abstracción matemática). La interpretabilidad en cascada: se puede leer la lógica como transformaciones simbólicas. Precisión OOD 10^-2 (el error de capa 1 se magnifica en capa 2).

#### V250 — Spectral Hysteresis Neuron (Stateful Memory)
- **Qué se probó:** Neurona con memoria EMA de representaciones espectrales (Walsh) para detectar novedad (Delta-Encoding).
- **Setup:** MNIST, orden aleatorio vs clustered.
- **Resultado principal:** Aleatorio: 91.80% (paridad); **Clustered: 43.32% vs 50.96% baseline (-7.6%)**.
- **Hallazgo:** El "Filtro de Novedad" (restar memoria) es destructivo en clusters: al restar el 50% de la memoria reciente, la red "limpia" las características comunes de la clase actual. La memoria debería usarse para normalización dinámica o gating, no para restar.

#### V251 — Multiplicative Gating Experiment
- **Qué se probó:** Gating multiplicativo por neurona (y = (xW+b)⊙g) con pesos congelados.
- **Setup:** MNIST, MLP 2 capas, 10 épocas.
- **Resultado principal:** **77.02% con solo 522 params** (780x reducción); PEI 28.33 vs 17.46 del MLP.
- **Hallazgo:** Entrenar solo el gating de una proyección aleatoria congelada es viable y extremadamente barato. Añadir gating a un modelo totalmente entrenable no aporta (97.76% vs 97.93%).

#### V251b — Multiplicative Gating Sweep
- **Qué se probó:** Escalado de D (32 a 4096) en gating congelado.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **89.30% con 4,106 params (D=4096)**; PEI pico en D=1024.
- **Hallazgo:** Escalado log-lineal hasta D=1024, luego rendimientos decrecientes. Las "funciones base" de la inicialización aleatoria son suficientemente ricas para MNIST.

#### V251c — Deep Multiplicative Gating
- **Qué se probó:** 2 capas de proyecciones congeladas (784→H→H→10) con 2H+10 gates.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **91.68% (D=4096)**; +16.19 puntos en D=128.
- **Hallazgo:** La profundidad mejora significativamente (especialmente en dimensiones pequeñas). La jerarquía de no-linealidades aleatorias crea un espacio más separable.

#### V251d — LR Sweep on Multiplicative Gating
- **Qué se probó:** Sensibilidad al LR (1e-3 a 5e-2).
- **Setup:** MNIST, D=512/1024/2048.
- **Resultado principal:** LR 5e-3 a 1e-2 óptimo; 5e-2 alcanza >90% en 1 época (D=2048).
- **Hallazgo:** La arquitectura es sorprendentemente estable a LRs altos. El espacio de parámetros restringido tolera LRs agresivos.

#### V251e — Scheduled Multiplicative Gating
- **Qué se probó:** OneCycleLR con MaxLR 0.05.
- **Setup:** MNIST, D=1024/2048/4096.
- **Resultado principal:** **93.89% con 4,106 params (D=4096)**.
- **Hallazgo:** El scheduler permite usar LR pico alto para encontrar features temprano y enfriar para estabilizar. ~94% con 4k params es un orden de magnitud más eficiente que el entrenamiento estándar.

#### V251f — Gate Sparsity Analysis
- **Qué se probó:** Weight decay (1e-3) para forzar esparcidad de gates.
- **Setup:** MNIST, D=4096.
- **Resultado principal:** **88.49%** (-5.4% vs 93.89%); solo 7% de gates silenciados a nivel 0.001.
- **Hallazgo:** La inteligencia en redes congeladas es colectiva: miles de detectores débiles se combinan. Forzar esparcidad daña el rendimiento; el "ruido" del reservoir es funcionalmente relevante.

#### V251g — Round-Robin Layer Training
- **Qué se probó:** Actualizar todos los gates cada batch, pero los pesos en rotación (1 capa por batch).
- **Setup:** MNIST, 3 capas.
- **Resultado principal:** [TBD - resultados pendientes].
- **Hallazgo:** Reduce el coste de cómputo del optimizador y gradiente de pesos en ~66% por batch.

#### V251j — High-LR Warmup & Switching
- **Qué se probó:** Fase 1: warmup de gates (LR 0.05); Fase 2: refinamiento de pesos (LR 0.001, gates congelados).
- **Setup:** MNIST, 3 capas.
- **Resultado principal:** Fase 1: 88.53% (3 épocas); **Fase 2: 97.63%** (época 9).
- **Hallazgo:** El gating hace optimización "coarse-grained" que coloca la red en una región favorable. El cambio a pesos causa un salto de +7% sin inestabilidad. Cerca de SOTA actualizando solo 33% de pesos por paso.

#### V251k — Weight Decay Impact on Gating
- **Qué se probó:** Comparar convergencia con y sin weight decay.
- **Setup:** MNIST, D=4096.
- **Resultado principal:** WD=0: 93.78%; **WD=1e-3: 88.49% (-5.29%)**.
- **Hallazgo:** **NUNCA usar weight decay** en gating de pesos congelados. La red necesita alto rango dinámico (hasta 20x) para "esculpir" el reservoir. El WD silencia los votantes débiles que forman el consenso colectivo.

#### V251L — The Oligarchy Hypothesis
- **Qué se probó:** Inicialización de gates en 0.0 (Discovery Mode) con SiLU.
- **Setup:** MNIST, D=4096.
- **Resultado principal:** **94.27%** (supera al baseline 94.18%); ~1965 gates efectivos (PR).
- **Hallazgo:** El "Discovery Mode" (Init=0 + SiLU) es superior: fuerza a la red a activar solo features que reducen la loss. La oligarquía: ~1965 "reyes" de 4096 son suficientes (ratio 2.1x de compresión).

#### V253 — Frozen Ternary Weights + Float Gating
- **Qué se probó:** Pesos ternarios congelados {-1,0,1} + gates float, D=2048.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **94.74%** con 4,106 params aprendibles (5.8M congelados).
- **Hallazgo:** Las proyecciones ternarias aleatorias son suficientemente ricas. Efecto "self-scaling": el clasificador reduce sus gates a ~0.09 para estabilizar el softmax (auto-regularización emergente).

#### V254 — The Importance of Inhibition ({0,1} vs {-1,0,1})
- **Qué se probó:** Aislar la contribución de pesos negativos.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **Ternary: 94.7%**; **Binary {0,1}: 41.4%** (colapso).
- **Hallazgo:** Los pesos binarios tienen media positiva (0.5), causando explosión acumulativa de activaciones (loss inicial 357). La inhibición (negativos) es esencial para contraste y detección de bordes. Sin -1, la red solo hace "summation pooling".

#### V255 — Full Ternary Networks (Multiplication-Free)
- **Qué se probó:** Pesos ternarios congelados + gates ternarios aprendibles (STE).
- **Setup:** MNIST, 15 épocas.
- **Resultado principal:** **82.20%**; 56.4% sparsity en capa 1; **cero multiplicaciones** (solo add/sub/mask).
- **Hallazgo:** Inestabilidad discreta (STE noise: bit-flips causan saltos en loss). Emergencia de esparcidad sin L1 (confirma oligarquía). El gap de 12% vs gates float es el coste de cuantización extrema (aceptable para FPGAs).

#### V256 — Ternary Gated CNN (Spatial Efficiency)
- **Qué se probó:** Kernels 5x5 ternarios congelados + gates de canal float.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** **85.07% con 394 params** (128/256 filtros).
- **Hallazgo:** El prior convolucional (invarianza local) es mucho más compatible con features ternarias aleatorias que proyecciones de píxeles. La red "mina" primitivas visuales (bordes, esquinas) de ~1M filtros aleatorios. Trade-off: ahorro de memoria pero coste de cómputo en entrenamiento.

#### V257 — Ternary CNN with Global Average Pooling (GAP)
- **Qué se probó:** Reemplazar el FC aplanado por GAP.
- **Setup:** MNIST, 10 épocas, 394 params aprendibles, 824,960 congelados.
- **Resultado principal:** **83.40%**; PEI ~28.6; loss inicial 2.22 (vs 6.03 de V256).
- **Hallazgo:** GAP reduce la "materia oscura" (123k params congelados) y actúa como regularizador (filtra ruido de alta frecuencia). La arquitectura más "production-ready" para visión embebida ultra-low-power.

### Era PID/Control (v258–v274)

> **Tema:** aplicar control industrial (PID: Proportional-Integral-Derivative) a la optimización de redes neuronales. Se descubren regímenes de ganancia extrema (`Ki` alto) que actúan como "Super-Momentum", y estrategias de phase-shifting (cambio de fase) que combinan exploración de alta energía con refinamiento amortiguado.
> **Hito clave:** V261 logra **98.47%** MNIST (Ki=150); V273 logra **83.25%** CIFAR-10 (phase shift híbrido); V274 valida el trigger automático.

#### V258 — Spectrum-Gated Transformer (SGT)
- **Qué se probó:** Transformer "Vision-like" donde la Self-Attention es reemplazada por FWHT (Fast Walsh-Hadamard Transform) sobre parches, con pesos ternarios congelados y float gating.
- **Setup:** MNIST, 1,546 params aprendibles, 541,696 congelados, 10 épocas.
- **Resultado principal:** **71.20%**.
- **Hallazgo:** Mezcla espectral global (Hadamard) alcanza 71% sin atención aprendida. El prior local (CNN) es más eficiente para MNIST. La implementación vectorizada (precomputed Hadamard matrix) reduce el tiempo de 1h a ~150s/época (regla: "vectorización o muerte").

#### V259 — Residual SGT (RSGT)
- **Qué se probó:** Añadir residual connections + duplicar hidden_dim a 1024 para resolver plateaus de V258.
- **Setup:** MNIST, 3,082 params aprendibles, 2,127,872 congelados, 10 épocas.
- **Resultado principal:** **72.12%** (+1%).
- **Hallazgo:** Los residuales son necesarios en redes gated profundas (preservan señal de parche). Pero sin positional encodings, el escalado de capacidad no compensa la falta de conciencia geométrica.

#### V260 — High-Res Positional SGT (PSGT)
- **Qué se probó:** Parches 2x2 (256 tokens en vez de 64), Sin/Cos positional encodings, 4 bloques residuales, OneCycleLR MaxLR=0.01.
- **Setup:** MNIST, 1,290 params aprendibles, 396,800 congelados, 15 épocas.
- **Resultado principal:** **91.69%** (PEI ~29.4).
- **Hallazgo:** La resolución es el factor dominante. Los positional encodings son el catalizador (sin ellos, el modelo no entiende geometría). OneCycleLR permite discovery agresivo + refinamiento estable.

#### V261 — PID Optimizer (The Industrial Miracle)
- **Qué se probó:** Implementar un controlador PID (Kp/Ki/Kd) como optimizador, sustituyendo Adam en un MLP estándar.
- **Setup:** MNIST, 10 épocas, MLP estándar.
- **Resultado principal:** Adam: 97.85% (loss 0.0178); **PID (1,150,1): 98.47%** (loss 0.0027, 6.5x menor).
- **Hallazgo:** La ganancia integral alta (`Ki=150`) actúa como "inercia" que ignora ruido local. El derivativo (`Kd=1`) frena el overshoot. Dinámicas (velocidad/aceleración) > Estadísticas (media/varianza) para MNIST.

#### V265 — Universal PID Benchmark (The Industrial Standard)
- **Qué se probó:** Validar PID-100 en un MLP estándar con BatchNorm + Relaxed Clipping (max norm 10).
- **Setup:** MNIST, 10 épocas, MLP estándar + BatchNorm.
- **Resultado principal:** Adam+Clip10: 97.64% (oscila al final); **PID(1,100,1)+Clip10: 98.41%** (ascenso consistente).
- **Hallazgo:** La "Ley de Inercia" es general: high Ki actúa como low-pass filter que sigue la curvatura real del gradiente, ignorando ruido de mini-batch. Relaxed Clipping (10.0) da suficiente rango dinámico al PID.

#### V267 — Systematic PID Hyperparameter Sweep
- **Qué se probó:** Grid search 27 configs (Kp∈{0.1,1,10}, Ki∈{1,10,100}, Kd∈{0.1,1,10}) sobre MLP estándar.
- **Setup:** MNIST, 10 épocas.
- **Resultado principal:** Mejor: **PID(1,100,10): 98.27%**; Adam: 97.63%.
- **Hallazgo:** Ki=100 domina absolutamente (todos los configs con Ki=100 superan 98%). Kd alta mejora la estabilidad final. PID gana por estabilidad de convergencia profunda, no por velocidad.

#### V268 — CIFAR-10 PID Instability
- **Qué se probó:** Trasladar PID(1,100,10) (ganador de MNIST) a CIFAR-10 (dataset más ruidoso).
- **Setup:** CIFAR-10, MLP estándar, 10 épocas.
- **Resultado principal:** **Fracaso**: PID 74.05% vs Adam 75.11% (-0.59%); oscilaciones violentas (73%→69%→72%).
- **Hallazgo:** Ki=100 es demasiado alto para datos ruidosos: el optimizador acumula inercia en direcciones obsoletas. El PID requiere sintonización dependiente de la SNR de la tarea.

#### V269 — Extreme Integral Gain Discovery
- **Qué se probó:** Subir Ki drásticamente (hasta 500) en CIFAR-10, desafiando la intuición de control clásico.
- **Setup:** CIFAR-10, MLP estándar, 10 épocas.
- **Resultado principal:** Adam: 75.11%; PID(1,10,20): 68.39%; PID(1,100,1): 71.47%; **PID(1,500,1): 75.54%** (+0.43%).
- **Hallazgo:** Efecto "Cargo Train": Ki=500 es un low-pass filter tan fuerte que el ruido por-batch se vuelve negligible. Crea un "Super-Momentum" que navega valles estrechos sin desviarse. Kd alto (20) es dañino; Kd=1 es suficiente.

#### V271 — Breaking the 80% Barrier (Ki=1000)
- **Qué se probó:** Llevar Ki=1000 a una CNN ancha en Google Colab T4.
- **Setup:** CIFAR-10, StandardCNN (64-128 canales), 20 épocas.
- **Resultado principal:** Pico **80.41%** (época 9); final 78.35%. Llegada a 76.7% en época 4.
- **Hallazgo:** "Recuperación Elástica": el PID tiene memoria direccional. Tras un drop a 75.86% (época 11, overshoot), rebota a ~79.7% (época 16) porque la integral mantiene la dirección global correcta. Adam divergería o se estancaría.

#### V272 — The Damping Trade-off
- **Qué se probó:** Añadir Kd=10 para estabilizar las oscilaciones de V271.
- **Setup:** CIFAR-10, StandardCNN, Ki=1000, Kd=10.
- **Resultado principal:** **79.27%** (pico); estabilidad alta (±0.4% en fase final).
- **Hallazgo:** Coste del control: Kd alto previene drops pero también suprime los picos agresivos que maximizan accuracy. Frontera entre velocidad/precisión (V271) y fiabilidad/estabilidad (V272). Se necesita approach dinámico.

#### V273 — Phase Shift Discovery (Hybrid Drive)
- **Qué se probó:** Cambio de fase manual: Épocas 1-8 con Ki=1000/Kd=1 (exploración); Épocas 9-20 con Ki=100/Kd=20 (refinamiento).
- **Setup:** CIFAR-10, StandardCNN, 20 épocas.
- **Resultado principal:** **83.25%**; salto de **77.82% → 83.11% (+5.29pp)** en una sola época al cambiar fase.
- **Hallazgo:** El cambio de fase es superior al learning rate decay estándar: no solo reduce la velocidad, sino que cambia la **naturaleza** del movimiento (de inercial a amortiguado). "Instant Annealing": los pesos se congelan en la cuenca que orbitaban. Loss final 0.0061.

#### V274 — Autonomous Industrial Pilot
- **Qué se probó:** Hacer el phase shift automático: monitorear validation accuracy, disparar el cambio cuando no haya mejora en 2 épocas.
- **Setup:** CIFAR-10, StandardCNN, patience=2.
- **Resultado principal:** **82.71%**; trigger automático en época 7; salto **76.04% → 82.37% (+6.33pp)**.
- **Hallazgo:** El PID autónomo elimina la necesidad de scheduling manual de épocas. La "elasticidad del aprendizaje": un cambio de inercia/amortiguación puede causar saltos de accuracy masivos, sugiriendo que los modelos estándar están atrapados en órbitas subóptimas por falta de rango dinámico en sus optimizadores.

### Era Complejo/Fase (v275–v299)

> **Tema:** la transición hacia **neuronas de fase compleja** (interferencia de ondas), validación en lenguaje natural (Tiny Shakespeare), el descubrimiento de que el FFN está sobreparametrizado (NarrowFFN), y el salto a la **memoria Delta Phase** (V298) con capacidad asociativa O(N) que supera a Softmax.
> **Hito clave:** V298 logra **99.95%** MQAR en época 2 (O(N) vs O(N²)); V299 demuestra superioridad compleja vs real a iso-floats (+22.84pp); V282 logra PPL 5.35 con 19.2% de parámetros.

#### V275 — Complex-Valued MLP (Wave Interference)
- **Qué se probó:** Red neuronal de valores complejos (CVNN) con ModReLU para permitir interferencia constructiva/destructiva de fase.
- **Setup:** Comparación directa vs MLP real iso-parámetros en tareas de regresión y aproximación.
- **Resultado principal:** **PEI 1.9718** (complejo) vs 1.5530 (real).
- **Hallazgo:** La fase adicional proporciona un grado de libertad extra que el modelo real no tiene. La interferencia de ondas codifica representaciones más ricas con los mismos parámetros.

#### V276 — Complex MNIST FFT
- **Qué se probó:** Aplicar FFT 2D como entrada a una CVNN para procesar imágenes en dominio frecuencial complejo.
- **Setup:** MNIST, FFT 2D como preprocesado, CVNN con ~101K params.
- **Resultado principal:** **95.43%** (PEI 19.05 vs 18.46 real).
- **Hallazgo:** El dominio complejo permite representar magnitud y fase de la FFT, capturando información de estructura que el modelo real pierde al descartar la fase.

#### V277 — Complex Transformer (Hermitian Attention)
- **Qué se probó:** Atención con matrices Hermitian (Q·K^H) en lugar de producto punto real, para secuencias periódicas.
- **Setup:** Secuencias sintéticas periódicas, Transformer pequeño.
- **Resultado principal:** **Loss 0.6466** (4x mejor que real). PEI 2.06 vs 1.54.
- **Hallazgo:** La conjugación Hermiticiana permite al modelo aprender relaciones de fase que la atención real no puede expresar. Particularmente útil para datos con estructura cíclica.

#### V278 — Phase Spectral Mixer
- **Qué se probó:** Mezclador espectral con fase analítica e^(iφ) para codificar posición y contenido.
- **Setup:** Single Spike Half dataset, mezcla espectral.
- **Resultado principal:** **100% accuracy en Época 1**.
- **Hallazgo:** La fase es un codificador posicional extremadamente eficiente. Una sola época basta cuando la fase codifica tanto la identidad del token como su posición relativa.

#### V279 — Phase LM on Real Text (Tiny Shakespeare)
- **Qué se probó:** Comparar ComplexFFT vs Walsh PE en modelado de lenguaje sobre Tiny Shakespeare.
- **Setup:** Tiny Shakespeare, validación de perplexity.
- **Resultado principal:** **ComplexFFT_noPE: 0.0439** vs Walsh_PE: 0.1699 (4x mejor).
- **Hallazgo:** Sin positional encoding explícito, ComplexFFT aprende representaciones temporales intrínsecas superiores. La fase compleja naturalmente codifica orden secuencial.

#### V280 — Causal Phase LM (Leakage Detection)
- **Qué se probó:** Implementar causalidad en Phase LM con zero-padding.
- **Setup:** Causal masking con zero-padding, PPL evaluada.
- **Resultado principal:** **0.0171** (leakage detectado).
- **Hallazgo:** El zero-padding no es verdaderamente causal: la FFT ve información del futuro a través de los bordes. Requiere implementación causal estricta.

#### V281 — True Causal Phase LM
- **Qué se probó:** FFT causal verdadero (h[t>0]=0) para eliminar leakage.
- **Setup:** FFT causal estricta, evaluación de PPL.
- **Resultado principal:** **1.7222** (causal real).
- **Hallazgo:** La causalidad estricta elimina el leakage pero reduce la capacidad expresiva. Walsh no puede ser verdaderamente causal por su naturaleza global.

#### V282 — Ultimate Phase-nGPT
- **Qué se probó:** Combinar CausalFFT + NarrowFFN + nGPT (normalización hiperesférica).
- **Setup:** Tiny Shakespeare, 116,870 params (19.2% del baseline denso).
- **Resultado principal:** PPL **5.35** vs 4.77 baseline.
- **Hallazgo:** La compresión extrema (19.2% params) es viable pero con trade-off en PPL. nGPT + phase es una arquitectura prometedora para edge.

#### V283 — Matrix-Free Phase-nGPT
- **Qué se probó:** Reemplazar todas las proyecciones lineales densas por WalshLinear (sintetizadas, sin matrices almacenadas).
- **Setup:** Tiny Shakespeare, 42,764 params, síntesis Walsh.
- **Resultado principal:** Loss **1.6581** (supera al modelo denso de 116K).
- **Hallazgo:** Matrix-free es posible y competitivo. La síntesis Walsh elimina el cuello de botella de memoria sin sacrificar capacidad.

#### V284 — Spherical Loss & Phase Reg
- **Qué se probó:** Temperatura τ aprendible + regularización de continuidad de fase en nGPT.
- **Setup:** nGPT con τ aprendible, regularización de fase.
- **Resultado principal:** **1.7664** con 24K params. τ escala de 10→43.5.
- **Hallazgo:** La temperatura aprende a enfriar la distribución de atención. La regularización de continuidad previene saltos abruptos de fase que rompen la diferenciabilidad.

#### V285 — Fourier Hippocampus (Contexto Infinito O(1))
- **Qué se probó:** Memoria hipocampal con K_mem=16 frecuencias bajas por capa, comprimiendo contexto en O(1) RAM.
- **Setup:** Contexto largo, 16 frecuencias bajas por capa, streaming.
- **Resultado principal:** **99.8% exact match en Época 3**.
- **Hallazgo:** Contexto infinito con estado constante es posible si el modelo aprende a comprimir en bajas frecuencias. La fase Fourier preserva el orden temporal sin crecer en memoria.

#### V286 — Poincaré Attention
- **Qué se probó:** Mover la atención al disco de Poincaré (geometría hiperbólica) con Soft-Tanh.
- **Setup:** Disco de Poincaré, d=4, atención con curvatura.
- **Resultado principal:** **38.37%** (vs 35.35% Euclidiana).
- **Hallazgo:** La geometría hiperbólica proporciona un espacio más rico para jerarquías naturales. Mejora marginal pero en una dirección contraintuitiva (menos dimensión, más estructura).

#### V287 — Conformal Optics
- **Qué se probó:** Interpretar pesos como proyecciones de textura compleja bajo transformación conforme.
- **Setup:** Pesos como mapas holomorfos, regularización de suavidad.
- **Resultado principal:** Concepto teórico validado; experimento de concepto.
- **Hallazgo:** El holomorfismo induce suavidad automáticamente. Aún pendiente de aplicación práctica en arquitecturas profundas.

#### V288 — Spectral Compression Zero-Shot (GPT-2)
- **Qué se probó:** Aplicar DCT como "BMP a JPG" a GPT-2: podar coeficientes DCT sin fine-tuning.
- **Setup:** GPT-2, poda DCT 10%, evaluación zero-shot.
- **Resultado principal:** Poda 10% → **95.41 PPL** (vs 89.58 sin podar).
- **Hallazgo:** La poda DCT es menos efectiva que la Walsh para LLMs. La DCT asume continuidad que los pesos de atención no tienen.

#### V289 — Spectral Quantization (DCT canal-por-canal)
- **Qué se probó:** Cuantización DCT 2-bit por canal vs RTN 2-bit en GPT-2.
- **Setup:** GPT-2, cuantización 2-bit, evaluación PPL.
- **Resultado principal:** RTN 2-bit: 2710 PPL; DCT 2-bit: mejora marginal.
- **Hallazgo:** La cuantización espectral en dominio DCT no rescata la pérdida de precisión. El dominio DCT no es universalmente mejor para cuantización.

#### V290 — Permutation Spectral (TSP + DCT)
- **Qué se probó:** Permutar canales MLP por TSP greedy → DCT → cuantización.
- **Setup:** GPT-2, permutación de canales, DCT post-orden.
- **Resultado principal:** Permutación preserva PPL (±1.71e-5). DCT post-orden mejora compresión.
- **Hallazgo:** El orden espacial de los canales importa. Reordenar por similitud (TSP) hace la señal más comprimible en DCT.

#### V291 — Oligarchy Validation (Cross-Dataset)
- **Qué se probó:** Validar la hipótesis de oligarquía (pocos gates dominantes) en Fashion-MNIST y CIFAR-10.
- **Setup:** 3 capas, D=4096, Fashion-MNIST y CIFAR-10.
- **Resultado principal:** ~51% gates activos (D=4096).
- **Hallazgo:** La oligarquía es robusta across datasets. La mitad de los gates pueden silenciarse sin pérdida significativa.

#### V292 — MQAR Spectral Mixer
- **Qué se probó:** CausalComplexFFT para MQAR (associative recall).
- **Setup:** MQAR benchmark, complejo causal.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Error de arnés detectado post-mortem. Resultado no fiable.

#### V293 — Holographic Phase Recall
- **Qué se probó:** HRR (Holographic Reduced Representations) para MQAR.
- **Setup:** MQAR, HRR con fase.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Mismo error de arnés. Resultado descartado.

#### V294 — Multihead Holographic
- **Qué se probó:** Holographic multicabeza para MQAR.
- **Setup:** MQAR, multi-cabeza holográfica.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Error sistemático en eval harness afectó múltiples experimentos.

#### V295 — Phase Sharpener
- **Qué se probó:** Añadir armónicos 2θ, 4θ, 8θ para agudizar la fase.
- **Setup:** Fase con armónicos múltiples.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Mismo error de arnés.

#### V296 — Causal Norm
- **Qué se probó:** Normalización estilo RetNet/RWKV en dominio complejo.
- **Setup:** Normalización causal, FFT.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Error sistemático.

#### V297 — Phase Softmax
- **Qué se probó:** Softmax selectivo por contenido en dominio de fase.
- **Setup:** Softmax con gate de fase.
- **Resultado principal:** **INVÁLIDO** (harness error pre-v298).
- **Hallazgo:** Mismo error.

#### V298 — Delta Phase MQAR (ANCLA)
- **Qué se probó:** Regla Delta matricial + fase compleja para memoria asociativa O(N).
- **Setup:** MQAR, ComplexDeltaPhase, 99.95% en época 2. Arnés corregido post-invalidación.
- **Resultado principal:** **99.95% accuracy en Época 2** (O(N) vs O(N²) de Softmax).
- **Hallazgo:** HITO. La Regla Delta en matriz compleja con fase resuelve MQAR casi perfectamente en 2 épocas. La fase compleja es el secreto: O(N) inference con calidad O(N²). Arnés corregido valida resultados previos inválidos.

#### V299 — Capacity Frontier (Iso-Floats)
- **Qué se probó:** Comparar ComplexDeltaPhase vs RealDeltaNet iso-floats en capacidad asociativa.
- **Setup:** d_k ∈ {32,64,128}, num_pairs 32–256, 5 semillas.
- **Resultado principal:** **95.98% a 64 pares** (Complejo) vs 73.14% (Real). +22.84pp.
- **Hallazgo:** Superioridad compleja demostrada a iso-floats. La geometría de fase (S¹) produce matrices de Gram mejor condicionadas que el espacio real. La ventaja no es capacidad bruta, es condicionamiento.

#### V300 — Capacity Scaling (En ejecución)
- **Qué se probó:** Barrido sistemático de d_k ∈ {32,64,128} vs num_pairs 32–256.
- **Setup:** Complejo vs Real vs Softmax MHA, LR sweep por brazo, 5 semillas.
- **Resultado principal:** **En ejecución**. Mide frontier de capacity completa.
- **Hallazgo:** Pendiente de finalización. Este experimento valida la escalabilidad de V299 y busca el punto de ruptura donde Softmax MHA deja de ser viable.

### Era v300+ — Roadmap Post-V300

> **Tema:** experimentos planificados tras V300 para escalar la memoria Delta Phase a lenguaje natural y producción.
> **Hito clave:** V304 (port a tiny-thinker) es el experimento crítico que valida transferencia a lenguaje real.

#### V301 — Phase Softmax Kernel
- **Qué se probó:** Kernel `exp(cos(Δθ)/τ)` vía expansión de Bessel para aproximar softmax en O(N).
- **Setup:** Expansión Bessel N_terms=4–8, truncamiento, MQAR.
- **Resultado principal:** **Planificado**.
- **Hallazgo:** Teóricamente prometedor pero con riesgo de truncamiento. La aproximación requiere trade-off entre términos de Bessel y d_k.

#### V302 — Dynamic State Decay / LRU
- **Qué se probó:** Decay dinámico λ_t aprendido por token para memoria con olvido controlado.
- **Setup:** λ_t = σ(λ_proj(x_t)), variantes global y por valor propio.
- **Resultado principal:** **Planificado**.
- **Hallazgo:** Gated DeltaNet ya publicado (Yang et al. 2024). V302 lo reimplementaría. Valor: validación de olvido controlado en memoria asociativa.

#### V303 — Multi-Head Specialization (Dual Memory)
- **Qué se probó:** Cabezas especializadas: recientes (λ≈1, d_k pequeño) vs globales (λ<1, d_k grande).
- **Setup:** H=4/8, routing por attention ligera.
- **Resultado principal:** **Planificado**.
- **Hallazgo:** Familia de arquitecturas ya explorada (Griffin/Hawk, Jamba/Zamba). Confirmaría tendencia conocida.

#### V304 — Port a tiny-thinker V12 (Validación en Lenguaje Natural)
- **Qué se probó:** Reemplazar StatefulComplexFFTMixer por ComplexDeltaPhaseHolographicBlock en modelo real.
- **Setup:** TinyStories, d_model=1024, H=8, d_k=128, L=8.
- **Resultado principal:** **Planificado** (CRÍTICO).
- **Hallazgo:** El único experimento que puede validar si la fase compleja transfiere a lenguaje natural. Criterio: Val loss ≤ 4.15 + MQAR > 90% + 2× speedup L>1024.

#### V305 — Spectral Quantization 4-bit
- **Qué se probó:** Cuantización espectral DCT/Walsh 8-bit bajas frecuencias + 4-bit altas.
- **Setup:** Matriz M ∈ ℂ^{d_k×d_k}, cuantización post-entrenamiento.
- **Resultado principal:** **Planificado**.
- **Hallazgo:** Baseline V289 es débil (1 semilla, PPL ~0.1). Requiere replicación antes de considerar baseline válido.

#### V306 — TSP Permutation + DCT para Cores Walsh
- **Qué se probó:** Resolver TSP greedy sobre núcleos Walsh entrenados → DCT → cuantización 4-bit.
- **Setup:** Núcleos Walsh de tiny-thinker, permutación por similitud coseno.
- **Resultado principal:** **Planificado**.
- **Hallazgo:** Mejora de compresibilidad de núcleos espectrales. Depende de V305 para baseline de cuantización.

---

## Metodología

- **Fuente:** archivos `docs/findings_v*.md`
- **Formato por experimento:** qué se probó, setup, resultados, conclusiones clave