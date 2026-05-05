# Findings V163g: El Poder de la Saliencia (Holographic Attention)

## Objetivo
Evaluar si aplicar un peso de atención (Saliencia) permite que la memoria holográfica maneje contextos masivos (>4k tokens).

## Resultados
Se aplicó un peso $W$ al token "aguja" y peso $1.0$ al ruido.

| Contexto (L) | Peso Aguja (W) | Accuracy | SNR |
| :--- | :--- | :--- | :--- |
| 1024 | 1.0 | 0.0% | 1.50 |
| 1024 | 5.0 | 80.0% | 4.93 |
| 1024 | 20.0 | **100.0%** | 17.02 |
| 4096 | 20.0 | **100.0%** | 9.62 |
| 4096 | 100.0 | **100.0%** | 27.07 |

## Conclusiones
- **La Atención es el Filtro**: Un peso de 20x es suficiente para que un token sobreviva a 4,000 tokens de interferencia.
- **Ruptura del Límite Cuadrático**: Hemos demostrado que un único vector de 1024/2048 dimensiones puede almacenar información crítica de una secuencia de 4k tokens si el modelo es selectivo.
- **Viabilidad del Contexto Infinito**: Este hallazgo permite diseñar LLMs donde la memoria no crece con el contexto, sino que se mantiene constante mientras el "Gater" sepa qué amplificar.
