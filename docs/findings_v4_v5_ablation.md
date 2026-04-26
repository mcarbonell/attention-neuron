# Findings: Ablation Study - Pure Multiplicative vs Pure Additive

## 1. Experimento

Se han implementado y evaluado dos variantes extremas de la arquitectura Attention Neuron para aislar la fuente de su capacidad de aprendizaje:

- **V4 (Pure Multiplicative)**: `W_eff = W_init * M + sin(bias)`
- **V5 (Pure Additive)**: `W_eff = W_init + A + sin(bias)`

Ambos modelos han sido entrenados en MNIST durante 10 épocas usando el optimizador Adam con una configuración idéntica (`rank=2`, `mask_prob=0.5`).

## 2. Resultados

| Variante | Fórmula Principal | Accuracy (10 Epochs, Rank=2) |
| :--- | :--- | :--- |
| **V4 (Pure Multiplicative)** | `W_init * M` | **86.64%** |
| **V5 (Pure Additive)** | `W_init + A` | 42.6% |

## 3. Conclusiones

Este experimento constituye una **prueba del algodón** fundamental para la tesis de la arquitectura:

1. **La corrección aditiva es insuficiente**: Una corrección aditiva de muy bajo rango (`rank=2`) sumada a un sustrato aleatorio congelado no tiene la capacidad expresiva necesaria para resolver el problema de forma eficiente (42.6%). Esto demuestra que el modelo completo *no* funciona simplemente por un efecto "LoRA sobre ruido".
2. **El gating multiplicativo es el motor principal**: La versión puramente multiplicativa logra extraer características altamente útiles del sustrato aleatorio (86.64%), demostrando que "apagar, encender o modular" las conexiones aleatorias fijas (los "ground tremors") contiene la mayor parte del poder representacional de la red.

## 4. Próximos Pasos

Habiendo demostrado que la fuerza reside en la modulación, el siguiente paso (V1: Residual Attention Neuron) probará la formulación `W_eff = W_init + W_init * M + A` para comprobar si un planteamiento residual ayuda a estabilizar el entrenamiento en las primeras épocas manteniendo la expresividad descubierta aquí.