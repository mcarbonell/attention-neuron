# Findings V163h: Compresión de Frases en el Holograma

## Objetivo
Validar la recuperación de secuencias ordenadas (frases) dentro de un holograma saturado de ruido.

## Resultados
Se insertó una frase de **8 tokens** con peso 30x y 2,000 tokens de ruido con peso 1x.

- **Fidelidad Media de Frase**: **100.0%**
- **Recuperación Posicional**: El mecanismo de `Roll` permitió interrogar cada posición (0-7) y recuperar el token exacto en todos los casos.

## Conclusiones
- **Preservación del Orden**: El desplazamiento circular (`Roll`) es una forma extremadamente robusta de codificar posición. No hay confusión entre "A luego B" y "B luego A".
- **Efecto de Saliencia Secuencial**: Al dar peso a una frase completa, esta se convierte en una "huella" dominante en el holograma que el modelo puede leer secuencialmente.
- **Eficiencia de Almacenamiento**: Hemos comprimido una idea secuencial compleja en un solo vector de 2048 dimensiones, rodeado de ruido masivo, sin pérdida de fidelidad.
