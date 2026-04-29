# Findings V103 - V106: The Haar Wavelet Era

## Resumen de la Investigación
Tras el éxito de las neuronas geométricas (Conos 2D, V101), exploramos la hipótesis de que las **Wavelets de Haar** podrían proporcionar una base de representación aún más eficiente al capturar naturalmente la localización espacial y la orientación de los bordes (Horizontal, Vertical, Diagonal).

## Evolución de los Prototipos

### V103: Haar Fixed Grid (3.6k parámetros)
- **Concepto:** Transformada de Haar 2D de 5 niveles con promediado en cuadrículas fijas (4x4, 2x2).
- **Resultado:** **93.08%** en MNIST.
- **Lección:** Confirmó que los bordes de Haar son "características SOTA" instantáneas (85% en Epoch 1), pero la cuadrícula fija emborrona detalles finos.

### V104: Selective Haar Low-Rank (Rank=2, 2.6k parámetros)
- **Concepto:** Selección dinámica de 1,024 coeficientes mediante una matriz de bajo rango (U @ V).
- **Resultado:** **78.10%** (Fallo).
- **Lección:** El Rank=2 creó un cuello de botella extremo. La red no tenía suficiente "vocabulario visual" para distinguir los 10 dígitos.

### V105: Haar Funnel (3.1k parámetros)
- **Concepto:** Embudo extremo de 1,024 entradas a solo 3 neuronas ocultas.
- **Resultado:** **39.67%** (Fallo).
- **Lección:** 3 neuronas son insuficientes para comprimir la semántica de MNIST, incluso con bordes perfectos.

### V106: Haar Selective XL (12.6k parámetros) - RÉCORD DE EFICIENCIA
- **Concepto:** 128 neuronas ocultas con un Rank de selección de 8 y **BatchNorm Espectral**.
- **Resultado:** **96.20%** en MNIST.
- **Hito:** Logró un **94.77% en la Época 1**, superando a casi cualquier otra arquitectura del repositorio en velocidad de convergencia inicial.
- **Conclusión:** La combinación de la potencia de Haar para bordes y una selección de bajo rango flexible (Rank=8) permite reducir los parámetros en un 90% respecto a una red densa estándar.

## Análisis Técnico: ¿Por qué Haar?
1. **Localización Espacial:** A diferencia de Walsh (global), Haar sabe *dónde* está el trazo.
2. **Sensibilidad a la Orientación:** Detecta trazos verticales y horizontales por separado, algo vital para los números.
3. **BatchNorm Espectral:** Fue el "héroe silencioso" de la V106, equilibrando la energía entre las escalas gruesas y finas.

## Comparativa de Parámetros vs Precisión (MNIST)
| Modelo | Parámetros | Precisión | Eficiencia (Acc/Params) |
| :--- | :--- | :--- | :--- |
| Baseline MLP | ~100k | 98.0% | 0.00098 |
| V101 (Conos 2D) | 3.8k | 94.3% | 0.02481 |
| **V106 (Haar XL)** | **12.6k** | **96.2%** | **0.00763** |

*Nota: Aunque la V101 es más eficiente paramétricamente, la V106 es mucho más robusta y rápida en converger.*

**Scripts de Referencia:**
- `scratch/prototype_v103_haar_mnist.py`
- `scratch/prototype_v106_haar_xl.py`
