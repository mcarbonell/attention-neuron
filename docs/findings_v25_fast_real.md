# Findings: V25_FAST (The Quick Prism) - REAL DATA

## 1. El Experimento

Se probó una arquitectura simplificada de 3 capas convolucionales (64, 128, 256 canales) utilizando la mezcla plana de 4 sustratos aleatorios por canal y un Learning Rate fijo de 0.003.

**Configuración:**
- **Modelo**: FastPrismNet (3 capas Conv).
- **Mecánica**: 4 sustratos aleatorios por capa.
- **Optimización**: AdamW sin scheduler.

## 2. Resultados Reales

| Métrica | Valor |
| :--- | :--- |
| **Accuracy Época 1** | **48.71%** (Récord histórico de velocidad) |
| **Best Test Accuracy** | **67.74%** (Época 17) |
| **Estado Final (Época 20)**| 65.56% (Oscilación por falta de scheduler) |

## 3. Conclusiones

1.  **Potencia de Arranque**: La mezcla de 4 sustratos permite que la red "vea" casi de inmediato. Superar el 48% en la primera época es un hito de este repositorio.
2.  **Necesidad de Control**: El estancamiento y la oscilación final demuestran que, aunque la Alquimia de Sustratos es potente, requiere de un enfriamiento del Learning Rate (OneCycleLR) para consolidar los rasgos finos.
3.  **Simplicidad vs Profundidad**: 3 capas anchas son suficientes para alcanzar el 67% en minutos, pero para el 99% la profundidad de la ResNet es indispensable.

## 4. Próxima Fase (V26)

Fusión de la velocidad de arranque del Prisma Plano con la robustez estructural de la ResNet y un scheduler profesional.
