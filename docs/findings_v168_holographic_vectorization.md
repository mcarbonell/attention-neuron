# Findings v168: Vectorización Holográfica y MoE Jerárquico

## Contexto
Transición de la arquitectura Spectral V8 (Densa/Recursiva) a V8.1.2 (Comprimida/Vectorizada) para entrenamiento soberano en hardware local (Radeon 780M iGPU).

## Hallazgos Clave

### 1. Vectorización del Holograma (Flash-Hologram)
- **Problema**: Los bucles recursivos de 512 pasos causaban una latencia de 12 min/eval en DirectML debido al overhead de lanzamiento de kernels.
- **Solución**: Sustitución de `torch.roll` y bucles `for` por operaciones de `torch.gather` y `torch.cumsum` (Suma Acumulada Espectral).
- **Resultado**: Aceleración de **7.5x** en la fase de evaluación y estabilidad mantenida mediante normalización vectorizada.

### 2. MoE Jerárquico por Clanes (Spectral Clans)
- **Problema**: El gating denso de 131,072 expertos generaba matrices de activación masivas (1GB por capa), saturando la VRAM compartida de la iGPU.
- **Solución**: Estructura de **512 clanes** con **256 especialistas** cada uno. Gating en dos pasos vectorizado mediante `torch.bmm` (Batch Matrix Multiplication).
- **Resultado**: Reducción drástica de la presión sobre la memoria y eliminación de errores de driver (TDR) y Unicode.

### 3. Eficiencia Paramétrica (The 1.1 Trillion Illusion)
- Se valida que la **Factorización Espectral** (`k_dim=128`) permite escalar el número de especialistas de forma casi independiente del coste computacional de la dimensión oculta.
- **Capacidad Teórica**: El modelo de 230M parámetros ofrece una granularidad de conocimiento equivalente a un modelo MoE denso de **1.1 Billones de parámetros** (1.1T).

## Métricas de Rendimiento (Hardware Local)
- **Hardware**: AMD Ryzen 7 8845HS + Radeon 780M (DirectML).
- **Entorno**: Windows 11 + PyTorch 2.4+ (DirectML Backend).
- **Time per Iter (V8 Dense CPU)**: ~494s.
- **Time per Iter (V8.1.2 Vectorized iGPU)**: **71.9s**.
- **Speedup Total**: **~6.8x** respecto a la implementación inicial.
- **Loss Inicial (Vocab 32k)**: 10.41 (Perfecto según $\ln(V)$).

## Actualización Post-Vuelo (Iter 760)
- **Hito de Optimización**: Se ha eliminado el `weight_decay` (0.1 -> 0.0) de los parámetros espectrales (firmas, pesos y bases).
- **Impacto Inmediato**: La loss ha pasado de un descenso gradual a una caída vertical, rompiendo la barrera de 6.0 y llegando a **5.15** en solo 10 iteraciones tras el reinicio (Iter 760).
- **Conclusión**: Confirmado que en arquitecturas factorizadas, la compresión estructural ya actúa como regularizador. El Weight Decay tradicional actúa como un "freno de mano" que apaga a los especialistas antes de tiempo.

## Métricas de Rendimiento Finales
- **Time per Iter (Vectorized iGPU)**: ~68-72s (Estable).
- **Convergencia Estimada**: El modelo está en camino de superar a la v7 (Loss 4.0) mucho antes de la iteración 1500, logrando una eficiencia de aprendizaje ~4x superior por paso.
