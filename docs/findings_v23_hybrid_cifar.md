# Findings: V23 (The Hybrid) - CIFAR-10 Experiment

## 1. Experimento: "Cerebro Plástico, Sensor Alquímico"

Tras observar el estancamiento de los MLPs puros en CIFAR-10 (V22), se propuso una arquitectura híbrida para determinar si el cuello de botella era la extracción de rasgos o la lógica de decisión.

**Configuración:**
- **Capa 1 (Sensor)**: Rosetta (4 sustratos aleatorios + sintonización por diales). **Congelada**.
- **Capas 2 y 3 (Cerebro)**: Capas lineales estándar al 100%. **Plásticas/Entrenables**.
- **Parámetros entrenables**: 2,452,490 (~29.2% de un MLP denso equivalente).
- **Pesos Congelados**: ~6,300,000.

## 2. Resultados

| Métrica | Valor |
| :--- | :--- |
| **Best Test Accuracy** | **62.51%** |
| **Época del Best** | 49/50 |
| **Tiempo de Entrenamiento**| 2988.8s (CPU) |
| **Comparativa V22 (MLP)** | **+5.79%** de mejora absoluta |

## 3. Conclusiones

1.  **Aceleración de Convergencia**: La V23 alcanzó el récord de la V22 (56.7%) en solo 10 épocas, demostrando que un cerebro plástico procesa la información del sensor Rosetta de forma mucho más eficiente que un sistema de modulación de bajo rango.
2.  **Validación del Sensor**: El hecho de que un cerebro plástico pueda clasificar CIFAR-10 al 62.5% usando una primera capa aleatoria fija confirma que la **Sintonía de Sustratos** genera una representación de rasgos (feature set) lo suficientemente rica para tareas de visión complejas.
3.  **Límite de la Invariancia**: A pesar de la plasticidad, la red sigue estando por debajo del Navigator (CNN). Esto reafirma que la arquitectura convolucional posee un sesgo inductivo que el MLP híbrido aún no puede replicar totalmente, incluso con sensores multicanal.

## 4. Próxima Fase (V24)

Traslado de la potencia "Alquimista" (Multi-sustrato) al dominio convolucional. El experimento **V24 "The Kaleidoscope"** buscará superar el 75% del Navigator mediante la sintonía de bibliotecas de kernels 3x3.
