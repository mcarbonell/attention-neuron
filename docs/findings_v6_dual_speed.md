# Findings: V6 (Dual-Speed Attention Neuron)

## 1. Experimento

Se ha evaluado la variante **V6 (Dual-Speed Attention Neuron)** partiendo de la formulación Residual (V1). El objetivo era probar la hipótesis de que la parte multiplicativa (que define la topología global) y la parte aditiva (que afina detalles) podrían interferirse si aprenden a la misma velocidad.

- **Configuración de LR**: 
  - Gating Multiplicativo ($M$): `lr = 0.001`
  - Corrección Aditiva ($A$) y Bias: `lr = 0.0001` (10x menor)
- **Fórmula Base**: `W_eff = W_init + W_init * M + A + sin(bias)`

Entrenamiento en MNIST (10 épocas, Adam, `rank=2`, `mask_prob=0.5`).

## 2. Resultados

| Variante | Configuración LR | Accuracy (10 Epochs) |
| :--- | :--- | :--- |
| **V1 (Residual)** | Uniforme (0.001) | **87.61%** |
| **V6 (Dual-Speed)**| Split (0.001 / 0.0001)| 86.75% |

## 3. Conclusiones

1. **Ralentización de la Convergencia**: Frenar la velocidad de aprendizaje de la corrección aditiva retrasa el arranque de la curva de aprendizaje (18% vs 24% en la época 1) y penaliza el resultado final a 10 épocas. 
2. **Sinergia vs Interferencia**: Los datos sugieren que la modulación topológica y el ajuste fino no se "estorban", sino que se benefician de actualizarse en la misma escala temporal. Adam parece gestionar bien el equilibrio de gradientes entre ambos componentes sin necesidad de "congelar" artificialmente ninguno de los dos flujos.

## 4. Próximos Pasos

Habiendo descartado mejoras sencillas vía activaciones no lineales (V2) o por velocidades de aprendizaje separadas (V6), la formulación base **V1 Residual** se consolida como la más limpia, robusta y eficiente. 

El próximo experimento natural sería evaluar la variante **V3 (Sparse Attention Neuron)** añadiendo regularización L1 para forzar una selección más interpretativa de características del sustrato.