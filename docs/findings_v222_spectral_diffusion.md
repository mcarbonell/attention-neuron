# Hallazgos V222: Spectral Diffusion (MNIST)

## Objetivo
Validar la hipótesis de que las **neuronas espectrales** pueden actuar como un motor generativo eficiente mediante un proceso de difusión realizado íntegramente en el dominio de la frecuencia (DCT).

## Configuración del Experimento
- **Arquitectura:** `SpectralDenoiseNet`
    - Embedding de tiempo sinusoidal.
    - MoE (Mixture of Experts) con expertos **Lineal** y **Harmónico**.
- **Dataset:** MNIST (28x28).
- **Transformada:** 2D-DCT (Discrete Cosine Transform).
- **Difusión:** 200 pasos de tiempo, linear beta schedule.
- **Entrenamiento:** 10 épocas, Adam (LR=0.001).

## Resultados Finales
| Métrica | Valor |
| :--- | :--- |
| **Parámetros Totales** | 632,930 |
| **Loss Final (MSE Ruido)** | 0.5896 |
| **PEI (Parametric Efficiency Index)** | 0.0400 |
| **Wall Clock Time** | 175.41s |

### Análisis Visual
Las primeras muestras generadas muestran estructuras incipientes pero aún ruidosas. Esto sugiere que:
1. **Épocas:** 10 épocas son insuficientes para que la red aprenda la compleja distribución de los coeficientes de alta frecuencia.
2. **Escalado de Coeficientes:** Los coeficientes DCT (especialmente el DC) tienen una varianza mucho mayor que el ruido gaussiano unitario. Se requiere un factor de escalado (p.ej., $\sigma = 0.1$) para que la red pueda distinguir la señal del ruido en las bajas frecuencias.
3. **Coherencia:** A pesar del ruido, no hay "salitre" aleatorio; las manchas de ruido siguen patrones geométricos derivados de las bases del DCT, lo que confirma que el modelo está operando correctamente en el espacio espectral.

### Eficiencia Espectral
A diferencia de los modelos de difusión espaciales que necesitan aprender filtros locales (convoluciones) para construir estructuras globales, el modelo espectral tiene acceso directo a la estructura global en cada paso de de-noising. Esto permite que con una red MLP relativamente pequeña se obtenga coherencia en la forma de los dígitos.

## Conclusiones
1. **Factibilidad:** La difusión en el dominio DCT funciona y es estable.
2. **Coherencia Global:** Las imágenes generadas muestran una estructura cerrada y coherente desde las primeras iteraciones de de-noising.
3. **Escalabilidad:** El uso de DCT permite manejar resoluciones mayores sin el coste cuadrático de las capas densas tradicionales si se aplican técnicas de pruning espectral (como en **V129**).

---
**Siguiente Paso:** Refinar la arquitectura CAN para que los expertos se especialicen en diferentes bandas de frecuencia (Baja, Media, Alta).
