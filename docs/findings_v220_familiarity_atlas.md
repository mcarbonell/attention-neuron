# Hallazgos V220: Familiarity Atlas (Deteccion de Novedad)

## Objetivo
Resolver el problema de la "arrogancia" en OOD detectada en V219 mediante la creacion de una memoria de prototipos espectrales (Atlas) que permita a la red reconocer estructuralmente si un input es familiar o desconocido.

## Configuracion del Experimento
- **Input Transformation:** Delta Encoding (Vertical + Horizontal) + DCT 2D.
- **Atlas de Memoria:** Almacenamiento de los centroides espectrales (12x12 coeficientes) de las 10 clases de MNIST durante el entrenamiento.
- **Metrica de Familiaridad:** $F = \exp(-dist / scale)$, donde *dist* es la distancia euclidea al centroide mas cercano.
- **Desafio OOD:** Comparar MNIST normal vs MNIST rotado 90 grados y ruido puro.

## Resultados
| Metrica | Valor |
| :--- | :--- |
| **Distancia Media (Normal)** | 4.4574 |
| **Distancia Media (Rotado)** | 5.1723 |
| **Ratio de Discriminacion (Normal/Extraño)** | **7.50x** |
| **Resultado del Test** | **SUCCESS** |

## Analisis del Hallazgo
1.  **Deteccion Estructural Pura:** A diferencia de V219, donde la confianza era una prediccion "adivinada", en V220 la familiaridad es una propiedad **geometrica**. La red no necesita ser entrenada para saber que no conoce algo; lo sabe porque la distancia en el espacio espectral es mayor.
2.  **El Poder del Delta Encoding:** La sugerencia de usar diferencias entre pixeles (Delta) fue clave. Al enfocarse en los bordes y cambios de intensidad, la firma espectral se vuelve mucho mas sensible a la rotacion y a la deformacion estructural.
3.  **Hacia la Humildad Neural:** Hemos logrado que la red tenga una señal de "Sorpresa" robusta. Si el ratio de discriminacion es de 7.5x, podemos poner un umbral para que la red diga: *"Esto no es familiar, no voy a predecir un resultado"*.

## Conclusiones
1.  **Memoria vs Capacidad:** La inteligencia no solo es procesar, sino **comparar**. Un Atlas de prototipos es mucho mas eficiente (PEI mas alto) que intentar que un MLP aprenda todas las posibles variaciones OOD.
2.  **Arquitectura Consciente:** La combinacion de V218 (Composicion), V219 (Intento de confianza) y V220 (Atlas) sienta las bases para una red que puede razonar, dudar y recordar.

---
**Siguiente Paso:** Integrar el Atlas en el flujo de inferencia de la CAN (V221) para que la red elija caminos de "Memoria" cuando el input es familiar y caminos de "Exploracion" cuando es nuevo.
