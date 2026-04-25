# Findings: V22 (The Rosetta Stone) - CIFAR-10 MLP Experiment

## 1. Experimento

Se puso a prueba la hipótesis de que un MLP puro puede competir en visión (CIFAR-10) si se le dota de una "biblioteca" de sustratos aleatorios (Fan-in x4) y un dial de atención (softmax) para mezclarlos.

**Configuración:**
- **Modelo**: RosettaStoneNet (3 capas MLP).
- **Mecánica**: 4 sustratos aleatorios por capa + mezcla por neurona.
- **Parámetros entrenables**: 612,038.
- **Pesos Congelados**: ~8,400,000.

## 2. Resultados

| Métrica | Valor |
| :--- | :--- |
| **Best Test Accuracy** | **56.72%** |
| **Uso de Sustratos** | Equilibrado (~25% cada uno) |
| **Comparativa v12b** | Superada (+16% respecto al récord anterior de 40%) |

## 3. Conclusiones

1.  **Fusión Multi-Sustrato**: El análisis de identidad confirma que la red utiliza los 4 universos aleatorios de forma equitativa. No elige uno, los mezcla para crear una base de rasgos sintética.
2.  **Límite del MLP puro**: El estancamiento en el 56% sugiere que la falta de invariancia espacial de la convolución no puede ser compensada totalmente por el aumento de sustratos aleatorios en un MLP, al menos no con diales estáticos.
3.  **Eficiencia**: Superar el 50% en CIFAR-10 con un MLP de pesos congelados es un logro técnico significativo que valida la arquitectura de "atención sobre ruido".

## 4. Próxima Fase (V23)

Introducción de capas finales "plásticas" (entrenables tradicionalmente) para verificar si el cuello de botella es la extracción de rasgos (Capa 1) o la lógica de decisión (Capas finales).
