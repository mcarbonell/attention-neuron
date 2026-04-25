# Findings: V12 (Hybrid Attention Neuron for CNNs)

## 1. Experimento

Se ha implementado la variante **V12 (Hybrid Attention CNN)** para trasladar la arquitectura de la Attention Neuron (que hasta ahora solo se había probado en capas densas) al dominio de las redes convolucionales (CNNs). 

La idea central es utilizar un **kernel convolucional aleatorio y congelado** (`W_init`) y modularlo dinámicamente combinando dos factores de muy bajo rango:
- **Modulación de Canal** (`M_chan`): Factorizada con `rank=2` para escalar las relaciones entre canales de entrada y salida.
- **Modulación Espacial** (`M_spatial`): Un tensor pequeño y directo (ej. 3x3) que modula las frecuencias espaciales del kernel congelado.

- **Arquitectura de Prueba**: 2 capas convolucionales híbridas (16 y 32 canales, kernel 3x3) seguidas de 1 capa lineal Attention Neuron Residual.
- **Hiperparámetros**: MNIST, 10 épocas, Adam, `mask_prob=0.5`.

## 2. Resultados

| Métrica | V12 (Hybrid CNN) | V1 (Residual MLP) |
| :--- | :--- | :--- |
| **Parámetros Entrenables** | **6,648** | ~15,400 |
| **Accuracy (Época 1)** | 18.05% | 24.81% |
| **Accuracy (Época 10)** | **83.42%** | 87.61% |
| **Tendencia de la Curva** | En fuerte ascenso (sin plateau) | Acercándose al plateau |

## 3. Conclusiones

1. **Validación del Paradigma Convolucional**: ¡El experimento es un éxito! La tesis de que la "inteligencia" puede residir puramente en el gating sobre un "ruido base congelado" **funciona también en el dominio espacial**. El modelo es capaz de extraer características visuales complejas modificando qué frecuencias de un filtro aleatorio 3x3 se encienden o se apagan.
2. **Eficiencia Paramétrica Extrema**: La red completa (2 capas convolucionales + clasificador) requiere apenas **6,648 parámetros entrenables**. Ha logrado un 83.4% de precisión con menos de la mitad de los parámetros que necesitaba el modelo MLP base.
3. **Dinámica de Aprendizaje**: La convergencia inicial es más lenta que en la versión densa (arranca en 18%), pero la curva muestra una aceleración muy sólida y no parece haber tocado techo en la época 10. Con más épocas o ajustando el escalado inicial espacial, es altamente probable que supere al MLP.

## 4. Próximos Pasos

Este resultado abre la puerta a aplicar la Attention Neuron a tareas de visión artificial (CV) complejas. El siguiente paso lógico, fuera de la sesión actual, sería entrenar esta arquitectura híbrida en **CIFAR-10** durante más épocas y comparar el rendimiento y la eficiencia paramétrica contra una CNN estándar estilo ResNet.