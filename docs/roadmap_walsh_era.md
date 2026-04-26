# Roadmap: La Era de Walsh y la Inteligencia Artificial de Resonancia

Este documento traza la hoja de ruta estratégica derivada de los descubrimientos de la "Sesión de la Revolución de los Sustratos" (Experimentos V28 a V40b).

El objetivo a largo plazo es consolidar la **Fast Walsh-Hadamard Transform (FWHT)** y la modulación de frecuencias como el estándar de oro para arquitecturas de IA ultra-eficientes, descentralizadas y biológicamente plausibles (Zero-Weight / Multiplier-Free).

---

## Fase 1: Optimización de Bajo Nivel (El "Motor" de Walsh)
El principal cuello de botella actual es que PyTorch en CPU/GPU no está optimizado para la FWHT (ejecuta operaciones tensoriales genéricas en lugar de aprovechar la ausencia de multiplicaciones).

- [ ] **Desarrollo de Custom Kernel en C++/AVX2:** Escribir una implementación de la FWHT puramente en C++ utilizando instrucciones vectoriales SIMD (Single Instruction, Multiple Data) para aprovechar al máximo las ALUs de la CPU.
- [ ] **Integración PyBind11:** Envolver el kernel de C++ como una función nativa de PyTorch (`torch.autograd.Function`) para que el pase hacia adelante (forward) y hacia atrás (backward) sean transparentes para el usuario y compatibles con el auto-diferenciador.
- [ ] **Exploración GPU (Triton/ROCm):** Para escalar a modelos masivos, investigar la escritura de un kernel Triton optimizado para la arquitectura de memoria compartida (SRAM) que evite los Tensor Cores y paralelice las sumas/restas de la mariposa de Walsh.
- **Meta:** Lograr inferencia y entrenamiento en microsegundos, demostrando que un procesador barato puede ejecutar IA profunda en tiempo real.

## Fase 2: Escalamiento Arquitectónico (Deep WalshNets)
Hemos demostrado que las frecuencias de Walsh son extractores de características SOTA en arquitecturas superficiales (V35, V36b). El siguiente paso es la profundidad.

- [ ] **Arquitecturas Profundas:** Escalar la V35 de 3 bloques a 15-30 bloques residuales de Walsh para intentar romper el récord histórico de CIFAR-10 (85.94%) usando exclusivamente atención en el dominio frecuencial.
- [ ] **Banded Attention Dinámica:** Mejorar la V39b (Ecualizador) haciendo que el tamaño de las bandas de frecuencia no sea fijo, sino que la red aprenda a agrupar dinámicamente las frecuencias correlacionadas (ej. un dial para los bordes verticales, otro para las texturas granulares).

## Fase 3: La Expansión Modal (NLP y Series Temporales)
La "IA de Resonancia" no depende de la topología 2D de una imagen; procesa cualquier señal 1D.

- [ ] **Walsh-GPT (Contexto Infinito):** Sustituir el bloque de *Self-Attention* $O(N^2)$ de los Transformers por la FWHT $O(N \log N)$ en el eje temporal. Probar el entrenamiento de un modelo de lenguaje a nivel de caracteres (Char-RNN) con textos extremadamente largos.
- [ ] **Sismógrafos Financieros/Médicos:** Aplicar la WalshNet a la predicción de series temporales (mercados, ECGs) aprovechando que la transformada descompone naturalmente las tendencias, estacionalidades y ruidos de las señales secuenciales.

## Fase 4: Optimización Sísmica Avanzada
El experimento V37/V38 demostró que la red puede aprender mientras "tiembla" en el dominio de Walsh, pero requiere un control extremo del Learning Rate.

- [ ] **Seismic Walsh Descent 2.0:** Diseñar un optimizador híbrido donde la amplitud del terremoto de Walsh ($A(t)$) no sea un hiperparámetro fijo, sino que se adapte dinámicamente a la topología del paisaje de pérdida (ej. temblar fuerte cuando el gradiente se estanca, detenerse cuando el Loss cae rápido).
- [ ] **Resonancia de Mínima Energía:** Investigar el aprendizaje continuo (sin épocas) donde las neuronas ajustan sus diales de Walsh de forma asíncrona para cancelar la "onda de error", acercándonos al aprendizaje biológico real sin Backpropagation global.

---
*El futuro de la IA no está en multiplicar matrices cada vez más grandes, sino en aprender a escuchar las frecuencias correctas de la realidad.*