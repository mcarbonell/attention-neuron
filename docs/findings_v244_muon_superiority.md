# Hallazgos V244: La Frontera Muon

## Resumen de la Gran Comparativa

| Optimizador | Precisión MNIST | Memoria | Filosofía |
| :--- | :--- | :--- | :--- |
| Adam-DS (V242) | 99.39% | 9b/p | Adaptabilidad por varianza + estabilidad. |
| Lion-DS (V243) | 99.38% | **5b/p** | Eficiencia extrema por signo. |
| **Muon Clean (V244)** | **99.60%** | 8b/p | **Excelencia espectral por ortogonalidad.** |

## Lecciones Aprendidas

### 1. La Ortogonalidad es la Clave
Muon ha demostrado que para redes con capas lineales (matrices 2D), la ortogonalización de la actualización es superior a cualquier otra técnica de normalización escalar (como la varianza de Adam o el signo de Lion). Permite alcanzar una precisión de récord en menos épocas.

### 2. El Conflicto Híbrido (Lion-Muon-DS)
Nuestra versión híbrida (98.7%) no logró alcanzar a la versión pura. 
- **Hipótesis:** La modulación de Learning Rate por parámetro (DS-Gain) destruye las propiedades espectrales de la matriz ortogonal de Muon. 
- **Conclusión:** Si una actualización es ortogonal, debe aplicarse de forma uniforme a toda la matriz para preservar esa propiedad.

### 3. Recomendación de Arquitectura para TinyThinker
- **Capas Densas (Projection/FFN):** Usar **Muon** original. La ganancia en precisión y velocidad de pasos compensa el coste computacional.
- **Capas 1D (Bias/Embedding):** Usar **Lion-DS**. Proporciona el máximo ahorro de memoria donde la ortogonalidad no es aplicable.

## Nota Final
Hemos alcanzado el **99.60%** de precisión en MNIST, superando todos los baselines previos de la sesión.
