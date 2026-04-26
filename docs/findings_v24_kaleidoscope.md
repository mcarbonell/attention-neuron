# Findings: V24 (The Kaleidoscope) - Efficiency Record

## 1. El Experimento

Se puso a prueba la arquitectura Kaleidoscope: una CNN donde cada canal de cada capa mezcla 4 universos de kernels 3x3 aleatorios fijos mediante un dial Softmax, aplicado con una modulación de rango reducido (`rank=16`).

**Configuración:**
- **Modelo**: KaleidoscopeNet (6 capas Conv).
- **Parámetros entrenables**: **64,062** (Mínimo histórico).
- **Sustratos**: 4 bibliotecas de kernels aleatorios por capa.
- **Entrenamiento**: 50 épocas, OneCycleLR.

## 2. Resultados

| Métrica | Valor |
| :--- | :--- |
| **Best Test Accuracy** | **75.18%** |
| **Parámetros** | 64,062 |
| **Eficiencia (Acc/Param)**| **1.17 % por cada mil parámetros** |
| **Comparativa V19** | -1.58% de accuracy con -46% de parámetros. |

## 3. Conclusiones

1.  **Victoria de la Alquimia**: La V24 rinde casi igual que la V19 pero con la mitad de parámetros. Esto demuestra que es más eficiente mezclar múltiples sustratos aleatorios que intentar modular uno solo con más rango.
2.  **Análisis de Fusión**: Las capas muestran un uso uniforme de los 4 sustratos (~25% cada uno), validando que el poder reside en la creación de una "base sintética de rasgos" mediante la combinación lineal de ruido.
3.  **Resiliencia**: El modelo es extremadamente estable a pesar de su pequeño tamaño, demostrando que el "sustrato rico" actúa como un potente regularizador.

## 4. Próxima Frontera (V25+)

Escalado masivo hacia el **99% de CIFAR-10** combinando arquitecturas residuales profundas con la sintonía dendrítica de sustratos aleatorios.
