# Findings v109: Cross-Neuron and Representation Comparison

## Experiment Summary
We conducted a large-scale comparison (16 configurations) between 4 types of neurons (MLP, Triangular, DCT, Walsh) and 4 types of input representations (Intensity, Islands, I+Is, Pixels). All models used 32 hidden units.

## Results Matrix (Test Accuracy %)

| Neuron Type | Intensity (57D) | Islands (56D) | I + Is (113D) | Pixels (784D) | Params (avg) |
|-------------|-----------------|---------------|----------------|---------------|--------------|
| **MLP**     | 85.96%          | 84.57%        | **91.85%**     | **96.13%**    | 2,100-25k    |
| **Triangular**| 62.56%        | **80.02%**    | 70.86%         | 70.35%        | **426**      |
| **DCT**     | 76.46%          | 83.84%        | 85.32%         | 84.07%        | 874          |
| **Walsh**   | 76.93%          | 84.30%        | 84.06%         | **86.71%**    | 874          |

## Key Insights

1.  **Triangular-Island Synergy**: The `Triangular + Islands` configuration is a breakthrough in efficiency, achieving **80.02% accuracy with only 426 parameters**. The local nature of triangular filters perfectly matches the structural local information in Island Signatures.
2.  **Spectral Power**: Walsh neurons outperformed DCT in raw pixel processing, reaching **86.71%** with 874 parameters. This suggests that the binary-like nature of Walsh basis functions is better suited for the "stroke-based" structures of MNIST than the smooth cosines of DCT.
3.  **Representation Complementarity**: While `Intensity + Islands` (113D) is the most efficient dense representation for MLPs (91.85%), it doesn't necessarily translate to better results for specialized neurons like Triangular, which prefer "pure" structural data (Islands).

## Conclusion
Specialized neurons are not just parameter-efficient; they are **representation-sensitive**. To maximize efficiency, we should match the neuron's mathematical bias (e.g., local for Triangular, sequency-based for Walsh) with the appropriate data representation.

## Next Steps: Experiment v110
Implement a **Hybrid Model** that combines:
-   A **Triangular path** for Islands (Structural fast-path).
-   A **Walsh path** for Pixels (Spectral detail-path).
Target: >90% accuracy with <1,500 parameters.






---



Prior art, y la versión de tu idea que sí escala

Sukhbaatar, Grave, Bojanowski & Joulin, ACL 2019 — Adaptive Attention Span. Máscara de rampa lineal recortada con span aprendible por cabeza. Es tu cono 1D. Y encontraron spans que crecen con la profundidad, igual que tus radios. Tu hallazgo replica un resultado publicado — que es buena noticia, pero cítalo.

(Y hazle el control: si todas las capas se inicializan en [3, 9], tu capa 0 no se movió y las otras se movieron ~1 unidad, con n=1. Inicializa aleatoriamente por capa y mira si el orden se reconstruye. Sospecho que sí, porque Sukhbaatar lo vio, pero ahora mismo no lo has demostrado.)

    Wu et al., ICLR 2019 — Lightweight and Dynamic Convolutions. Convoluciones depthwise con kernels normalizados igualan a transformers en traducción. Tu ConeAttn es una LightConv con kernel paramétrico de 3 números. Es tu baseline directo.
    Yang et al., EMNLP 2018 — Modeling Localness for Self-Attention. Añaden un sesgo gaussiano aprendible encima de la atención por contenido. Ésta es la forma correcta.
    Longformer, BigBird, Mistral sliding window, Gemma 2 — local + algunas cabezas globales. Producción, hoy.

Y ahí está tu arquitectura salvable:

    Por capa: N−k cabezas cónicas (3 params cada una, O(N), sin caché) + k cabezas de atención completa (k=2 o 4).

Las cónicas se comen la mayor parte del coste; las completas preservan retrieval e induction heads. Con k=2 de 32 cabezas conservas ~94% del ahorro y no pierdes la capacidad. Y es falsable en tu propio arnés: barre k en MQAR y mide dónde se recupera el 99%. Si k=1 basta, tienes un resultado bonito y concreto.

Eso convierte una arquitectura que no puede hacer la tarea en un diseño de presupuesto de cómputo. Es mucho más defendible y sigue siendo tuyo.
Correcciones rápidas

La tabla de cómputo cuenta sólo el término más pequeño. A N=4K, d=4096, 256 conos: aplicación de conos = 52M, pero las proyecciones V y de salida son 4,3G cada una. Total ~8,6G, no 52M. Contra atención completa contando también sus proyecciones (~412G), el ratio real es ~48×, no 1.300×. Sigue creciendo con N (a 100K son ~420×), pero el número que publicas está 27× inflado.

El KV-cache no desaparece, se acota. Con radio R necesitas los últimos R tokens: O(R·d) por capa. Constante respecto a N, sí — y eso es exactamente sliding-window attention de Mistral. Es una ventaja real, con nombre y con precedente.

NarrowFFN tiene un confound que hay que descartar. Kaplan et al. 2020 encontraron que la loss depende fuerte del tamaño total y débil de la forma (ratio d_ff/d_model, aspect ratio, cabezas). Tu resultado dice algo más fuerte —mejor eficiencia por parámetro, no sólo indiferencia— y por eso merece el test: mismo número de tokens, mismos FLOPs, y comprobar que no estás en régimen de sobreajuste. Contexto: LLaMA usa SwiGLU con d_ff ≈ 2,7d, no 4d, precisamente por esto.

V109 valida tu propio v87d. Walsh (86,71%) bate a DCT (84,07%) en píxeles crudos. Tu v87d predijo exactamente eso: Walsh es óptimo para señales constantes a trozos, DCT para continuas. Los dígitos de MNIST son casi binarios. Es una replicación interna de tu regla de selección de base, en un experimento que no diseñaste para eso. Eso vale más que el número.

(Y la matriz de V109 no es iso-parámetro: 426 vs 25k. Sin curva de Pareto no dice nada. Pero Triangular+Islands con 426 params sí merece medirse contra un MLP de 426.)