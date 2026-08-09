# Ficha Técnica ONE-002: Transformada Rápida de Walsh-Hadamard (FWHT / Spec-RAMA)

> 📚 **Familia:** 2. Neuronas Espectrales y Frecuenciales  
> 🏷️ **Etiqueta de Rigor:** [ANCLA] (Verificado en Proyecto Spec-RAMA)

---

## 1. Formulación Matemática & Origen Histórico
Basada en la matriz de Hadamard de Sylvester $H_n = H_1 \otimes H_{n-1}$ con $H_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$. 

La transformada `FWHT` reemplaza la multiplicación de matrices dense $O(N^2)$ por combinaciones lineales aditivas espectrales en el dominio binario/Walsh de complejidad $O(N \log_2 N)$:

$$y = \frac{1}{\sqrt{N}} H_N \cdot x$$

---

## 2. Hiperparámetros & Optimizador
- **Optimizador:** AdamW / Lion.
- **Weight Decay:** `weight_decay = 0.0` obligatorio (Regla de redes espectrales en `GEMINI.md`: las transformadas de Hadamard sufren degradación si se atenúa su espectro de frecuencia con norma L2).
- **Multiplicadores:** Cero multiplicaciones punto flotante en el núcleo de la transformada; solo sumas y restas.

---

## 3. Presupuesto Paramétrico & Intensidad Aritmética
- **Parámetros de la Transformada:** **0 Parámetros Entrenables** (sustrato estructural congelado).
- **Intensidad Aritmética:** Ultrabaja. Requiere $N \log_2 N$ sumas.
- **Factor de Compresión:** Permite redes 10x-100x más comprimidas al sustituir proyecciones lineales por mariposas de Hadamard.

---

## 4. Desempeño y Métrica Principal
- **Compresión Paramétrica:** Mantiene la capacidad de representación en clasificación y embeddings usando 10x menos parámetros que un MLP denso.

---

## 5. Dominio de Tarea & Benchmarks
- **Mezcla de Canales en LLMs:** Reemplazo de la proyecciones $W_q, W_k, W_v$ o proyecciones FFN.
- **Reducción de Dimensión y Hashing Espectral:** Clasificación rápida y hashing bi-nivel.

---

## 6. Perfil de Hardware & Latencia Real
- **Aceleración Hardware:** Máxima. En procesadores x86/ARM o FPGAs, se implementa con instrucciones SIMD AVX2/AVX-512 o NEON de adición/substracción paralela.
- **Vectorización Obligatoria:** Implementada mediante algoritmos mariposa in-place en PyTorch nativo.

---

## 7. Generalización Out-of-Distribution (OOD)
Alta robustez debido a la invariancia ortogonal de la matriz de Hadamard ($H_N^T H_N = N \cdot I_N$).

---

## 8. Interpretabilidad & Geometría del Espacio de Estados
- **Espacio de Sequency (Frecuencia Walsh):** Organiza la información según el número de cambios de signo (sequency) en lugar de frecuencias sinusoidales continuas.

---

## 9. Trazabilidad de Código & Scripts del Corpus
- **Proyecto Origen:** `spec-rama` ([README.md](file:///c:/Users/mrcm_/Local/proj/algorithms/spec-rama/README.md))

---

## 10. Amenazas a la Validez, Anomalías & Bugs Conocidos (⚠️)
- ⚠️ **Dimensión Potencia de 2:** Requiere que la dimensión $N$ sea estricta potencia de 2 ($N = 2^k$). Si no lo es, requiere padding con ceros.
