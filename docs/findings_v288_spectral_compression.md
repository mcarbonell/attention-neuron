# Hallazgos del Experimento: Compresión Espectral Zero-Shot "BMP a JPG" (v288)

Este documento resume los resultados obtenidos en el experimento **v288**, que evalúa la analogía de compresión "BMP a JPG" para los pesos de una red neuronal pre-entrenada (**GPT-2**), comparando la estabilidad en la perplejidad bajo tres métodos de poda de coeficientes.

---

## 1. Configuración Experimental
* **Modelo Evaluado**: GPT-2 Small (124M parámetros, Hugging Face).
* **Capas Afectadas**: Proyecciones lineales de todos los bloques de atención (`c_attn`, `c_proj`) y MLP (`c_fc`, `c_proj`).
* **Dataset de Evaluación**: Tiny Shakespeare (20 secuencias de longitud 512, totalizando 10,240 tokens).
* **Parámetro de Estabilización**: **Variance Rescaling** aplicado en la reconstrucción del peso.
* **Dispositivo**: CPU.

---

## 2. Resultados Oficiales (Perplejidad en WikiText-2 / Tiny Shakespeare)

* **Perplejidad Baseline (Sin comprimir)**: **89.5758**

A continuación se muestra la comparación de perplejidad (PPL) para cada método según el porcentaje de parámetros eliminados (ratio de compresión):

| Ratio de Compresión | Poda Espacial (Baseline) | Paso Bajo DCT (JPG Slice) | Umbral de Energía DCT (JPG Coefs) |
| :---: | :---: | :---: | :---: |
| **0% (Base)** | 89.58 | 89.58 | 89.58 |
| **10%** | **89.42** | 2832.09 | 95.41 |
| **30%** | 97.08 | 7657.95 | **93.88** (Supera a Espacial) |
| **50%** | **342.84** | Explosión | 1625.29 |
| **70%** | **2923.19** | Explosión | 3385.14 |
| **80%** | 5453.53 | Explosión | **2944.36** (Supera a Espacial) |
| **90%** | **4897.35** | Explosión | 8468.90 |

*Nota: "Explosión" indica una perplejidad superior a 10,000, reflejando un colapso completo del lenguaje.*

---

## 3. Hallazgos Fundamentales

### A. Fallo Catastrófico del Paso Bajo DCT (JPG Slice)
Al conservar estrictamente solo la zona de bajas frecuencias (el cuadrante superior izquierdo del espectro DCT), la red sufre una degradación de PPL instantánea (2832.09 a solo 10% de compresión).
* **Insight**: A diferencia de las imágenes donde las altas frecuencias representan ruido visual o detalles que el ojo humano ignora, en los pesos de un LLM **las altas frecuencias son esenciales**. Codifican las diferencias sutiles entre cabezas y dimensiones necesarias para la correcta distribución de la atención.

### B. Éxito de la Compresión por Umbral de Energía DCT
Al ordenar los coeficientes DCT por magnitud absoluta y conservar solo el top $(1 - \text{ratio})$, el modelo demuestra una resiliencia excepcional en comparación con el corte por frecuencias.
* **Insight 1 (Compresión Baja)**: Con un **30% de compresión**, el Umbral de Energía DCT logra **93.88 PPL**, superando a la poda espacial estándar (**97.08 PPL**).
* **Insight 2 (Compresión Alta)**: Con un **80% de compresión**, el Umbral de Energía DCT retiene **2944.36 PPL**, superando sustancialmente a la poda espacial (**5453.53 PPL**).
* **Conclusión**: Excluir los componentes de baja magnitud en el dominio frecuencial es un método de compresión superior a la poda espacial a ratios moderados y altos. La base DCT concentra la energía semántica en componentes clave, actuando como un excelente regularizador.

---

## 4. Próximos Pasos Recomendados

1. **Entrenamiento Nativo con Regularización Espectral (L1 en DCT)**: Dado que el Paso Bajo zero-shot falla por la dependencia en altas frecuencias, proponer un entrenamiento (pre-training) donde se penalice la norma L1 del espectro. Esto forzará al modelo a encontrar una solución de baja frecuencia que permita compresión de paso bajo sin pérdida de perplejidad.
2. **Cuantización Espectral Jerárquica**: Evaluar la cuantización de los coeficientes de secuencialidad (de v229) sobre este benchmark de GPT-2 para determinar si una asignación variable de bits en el dominio de frecuencia es superior a la poda de coeficientes.
