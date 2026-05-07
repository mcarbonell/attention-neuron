# Findings V250: Spectral Hysteresis Neuron (Stateful Memory)

## Resumen del Experimento
Se implementó una arquitectura de neurona única con **Histéresis Espectral (HSN)** que mantiene una memoria (EMA) de las representaciones espectrales (Walsh-Hadamard) de los inputs recientes. El objetivo era validar si "recordar" el pasado inmediato ayuda a la red a procesar el presente mediante la detección de novedad (Delta-Encoding).

## Métricas Clave

| Métrica | Baseline (Random) | HSN (Random) | Baseline (Clustered) | HSN (Clustered) |
| :--- | :--- | :--- | :--- | :--- |
| **Final Accuracy** | 92.08% | 91.80% | 50.96% | 43.32% |
| **PEI** | 22.96 | 22.89 | 12.71 | 10.80 |
| **Wall Clock Time** | 52.2s | 50.4s | 49.1s | 53.7s |
| **Overhead** | Bajo | Moderado | Bajo | Moderado |

## Hallazgos Críticos

1. **Estabilidad en Datos i.i.d.**: En el orden aleatorio estándar (MNIST normal), la memoria no degrada significativamente el rendimiento (-0.28%). Esto indica que la red puede tolerar un estado interno persistente sin entrar en caos.
2. **El "Filtro de Novedad" es Destructivo en Clusters**: En el escenario `Clustered Order` (donde los ejemplos vienen agrupados por clase), la red HSN sufrió una caída mayor que el baseline (-7.6% respecto al baseline). 
    - **Razón**: Al restar el 50% de la memoria reciente ($\beta=0.5$), la red está "limpiando" las características comunes de la clase actual. En un flujo de "ceros", la red intenta buscar lo que *no* es un cero, perdiendo la señal principal.
3. **Eficiencia en el Arranque**: En las primeras iteraciones de cada cluster, la HSN mostró una pérdida ligeramente menor (2.45 vs 2.69), sugiriendo una adaptación rápida que luego se vuelve contraproducente por la supresión excesiva de señal.

## Conclusiones y Próximos Pasos
La idea de memoria tiene sentido, pero la **resta de memoria** (novelty focus) solo es útil cuando el entorno es altamente redundante y buscamos anomalías. Para clasificación general, la memoria debería usarse para:
- **Normalización Dinámica**: Usar el estado interno para centrar los datos (Self-Centering).
- **Gating**: Activar diferentes "expertos" según el contexto recordado.

> [!TIP]
> Una mejora prometedora sería una **Memoria de Contraste**: En lugar de restar la memoria al input, usarla como entrada adicional para que la neurona decida qué información es relevante.

## Reproducibilidad
- **Archivo**: `scratch/prototype_v250_spectral_hysteresis.py`
- **Hardware**: CPU / DirectML
- **Semillas**: 1 (Determinista por orden de dataset)
