# Hallazgos V90c, V90d, V90e: Evolución de la Placa Analógica

## V90c: Placa Gated (MoE Intra-neuronal)
- **Concepto**: Introducción de un mecanismo de gating dinámico (`sigmoid`) que permite a cada neurona seleccionar qué agregador (SUM, VAR, L2, LSE, WALSH) usar basándose en la entrada.
- **Resultado**: 97.26% vs 96.92% (Baseline). Primera victoria sobre la red estándar.
- **Conclusión**: La especialización dinámica es superior a la mezcla estática.

## V90d: Spectral Tuning Plate
- **Concepto**: Sustitución del promedio global de Walsh por una **máscara espectral aprendible** por neurona. Cada neurona actúa como un sintonizador de bandas de frecuencia.
- **Resultado**: 97.27%. Mejora marginal sobre V90c, pero con mayor interpretabilidad (dispersión de máscaras de 0.26).
- **Conclusión**: Las neuronas aprovechan la capacidad de sintonizar frecuencias específicas.

## V90e: Holographic Resonator Plate [EL RÉCORD]
- **Concepto**: Implementación de **interferencia compleja** en el dominio de Walsh. Los pesos se dividen en componentes Reales e Imaginarios. La activación es la magnitud de la interferencia constructiva/destructiva entre el patrón de entrada y el almacenado.
- **Resultado**: **97.92%** (MNIST, 64 neuronas).
- **Eficiencia**: Supera al baseline por un 1% absoluto con la misma cantidad de neuronas, demostrando que la resonancia es una forma de representación mucho más densa que la suma lineal.
- **Teoría**: Se confirma la viabilidad de la "Memoria Holográfica" para tareas de clasificación.

## V90f: Fractal Resonator Plate
- **Concepto**: Análisis multiescala. Combina la resonancia holográfica global con el análisis espectral de parches locales (7x7).
- **Resultado**: 97.66%. 
- **Conclusión**: Aunque es muy potente y converge rápido, la complejidad adicional de los parches locales no superó la elegancia de la interferencia global de la V90e en este dataset específico.

---
*Estado actual: V90e se establece como la arquitectura de referencia para neuronas analógicas de alta densidad.*
