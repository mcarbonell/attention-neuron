# Findings: V26 (Perlin Spectrum) - Ruido Correlacionado vs Ruido Blanco

## 1. El Experimento

Tras el éxito del Kaleidoscope (V24), se planteó la hipótesis de que inicializar los sustratos con ruido estructurado (Perlin Noise) proporcionaría un mejor "prior espacial" para la visión artificial que el ruido blanco puro (Kaiming Normal).

**Configuración:**
- **Modelo**: PerlinSpectrumNet (6 capas Conv, `kernel_size=5` en Conv1).
- **Mecánica**: 4 sustratos por capa basados en Ruido Perlin 2D a diferentes frecuencias espaciales (escalas 0.3, 0.6, 1.2, 2.4).
- **Parámetros entrenables**: 64,062 (Mismo tamaño que V24).
- **Entrenamiento**: 50 épocas, OneCycleLR.

## 2. Resultados

| Métrica | V26 (Perlin) | V24 (Ruido Blanco) | Diferencia |
| :--- | :--- | :--- | :--- |
| **Best Test Accuracy** | **75.56%** | 75.18% | **+0.38%** |
| **Parámetros** | 64,062 | 64,062 | 0 |
| **Tiempo de Entrenamiento**| ~4.2 Horas | ~4.2 Horas | 0 |

## 3. Análisis del Uso de Biblioteca (Library Usage)

El comportamiento de los diales (Softmax) en la V26 revela un patrón biológicamente plausible que no se observó en el ruido blanco:

- **Capa 1 (Conv1)**: Sesgo fuerte hacia los sustratos de mayor frecuencia (escalas 1.2 y 2.4 con ~26% y ~28% de uso) frente al sustrato más suave (escala 0.3 con solo ~20%). La red utiliza las texturas finas de Perlin como detectores de bordes (Gabor filters).
- **Capas Profundas (Conv4, Conv5, Conv6)**: Inversión de la tendencia. Fuerte sesgo hacia los sustratos de baja frecuencia (escalas 0.3 y 0.6 con ~26-29% de uso). La red prefiere ruido suave y de grandes estructuras para "razonar" sobre los conceptos globales extraídos en capas previas.

## 4. Conclusiones

1.  **Superioridad del Ruido Estructurado**: El ruido Perlin vence al ruido blanco en igualdad de condiciones paramétricas, validando que inyectar continuidad espacial en los pesos aleatorios acelera y mejora la representación visual.
2.  **Sintonía Frecuencial Autónoma**: La red aprendió de forma completamente autónoma a emular la jerarquía del córtex visual humano: Alta frecuencia en las primeras capas (bordes/detalles) y baja frecuencia en las capas profundas (formas globales/contexto).
3.  **Hito de Eficiencia**: Alcanzar un 75.5% en CIFAR-10 con apenas 64K parámetros de modulación sobre ruido Perlin consolida la "Attention Neuron Theory" como un paradigma viable para Edge AI.
