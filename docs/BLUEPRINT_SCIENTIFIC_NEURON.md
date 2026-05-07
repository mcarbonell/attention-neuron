# Blueprint: The Scientific Neuron Architecture

## 1. Visión General
La **Neurona Científica** es una alternativa a la neurona tradicional (perceptrón) diseñada para el **descubrimiento de leyes matemáticas** y la **generalización estructural**. En lugar de aproximar funciones mediante la fuerza bruta de capas densas, utiliza un "menú" de funciones base y un mecanismo de selección agresivo.

## 2. Componentes Fundamentales

### A. Basis Augmentor (Expansión de Base)
Cada input $x$ se transforma en un vector de alta dimensión $B(x)$ que contiene primitivas matemáticas:
- **Polinómicas**: $x, x^2, x^3$
- **Trascendentales**: $\exp(x), \log(|x|)$
- **Trigonométricas**: $\sin(x), \cos(x)$
- **No Lineales**: $ReLU(x), \text{abs}(x), \text{sgn}(x)$
- **Interacciones**: $x_0 \cdot x_1$ (para inputs multidimensionales)

### B. Selección de Ley (L1 + Hard Thresholding)
Para garantizar la interpretabilidad y evitar la explosión en extrapolación (OOD):
1.  **Regularización L1**: Penaliza los pesos durante el entrenamiento para inducir dispersión (sparsity).
2.  **Hard Thresholding (Poda Agresiva)**: Tras el entrenamiento, cualquier peso $|w| < \tau$ (ej. $\tau = 0.05$) se pone a **CERO absoluto**. Esto elimina el ruido residual que causa inestabilidad en rangos no vistos.

## 3. Arquitectura Jerárquica (Deep Scientific)
La composición de funciones $g(f(x))$ se logra apilando capas de neuronas científicas.
-   **Capa 1**: Descubre leyes internas o transformaciones de coordenadas.
-   **Capa 2**: Aplica funciones externas sobre los resultados de la primera capa.
-   **Estabilidad**: Requiere *clamping* de señales y una inicialización pequeña para evitar que las bases explosivas (como `exp`) generen NaNs.

## 4. Ventajas vs MLP Tradicional

| Métrica | MLP (Denso) | Neurona Científica |
| :--- | :--- | :--- |
| **Parámetros** | Miles/Millones | Decenas/Cientos |
| **Interpretabilidad** | Caja Negra | Fórmula Matemática Pura |
| **Generalización OOD** | Pobre (Interpolador) | Perfecta (Descubridor) |
| **Mecanismo** | Ajuste Estadístico | Descubrimiento de Leyes |

## 5. Aplicaciones Recomendadas
-   **AI4Science**: Modelado de datos experimentales donde se busca una ley física.
-   **World Models**: Representación compacta de la física de un entorno para agentes de refuerzo.
-   **Hybrid LLMs**: Cabezas de razonamiento aritmético para grandes modelos de lenguaje.

---
*Este Blueprint consolida los experimentos v246-v249.*
