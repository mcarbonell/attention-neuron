# Blueprint: El LLM Espectral Ideal (Holographic Transformer)

Este documento define la arquitectura maestra basada en los hallazgos del proyecto. El objetivo es un modelo con el conocimiento de GPT-3, pero que funcione en una GPU integrada (Radeon 780M).

## 1. Capa de Comprensión (Spectral Embedding)
*   **Input**: Tokens discretos.
*   **Transformada**: Walsh-Hadamard (FWHT) inmediata.
*   **Firma**: Cada token se representa como una firma espectral normalizada en 1024 dimensiones.

## 2. El Motor de Conocimiento (Extreme MoE FFN)
*   **Ancho**: 131,072 Expertos Espectrales (Clanes).
*   **Gating**: Resonancia por producto escalar en el dominio de Walsh.
*   **Sparsity**: Solo los 16-32 expertos con mayor resonancia se activan (Top-K).
*   **Aprendizaje (Neurogénesis)**: [VALIDADO v167b] Los expertos nacen por "Sorpresa". Si un patrón no resuena (>0.8), se crea un nuevo experto instantáneamente. Esto permite aprendizaje One-Shot continuo.

## 3. La Memoria de Contexto (Spatiotemporal Hologram)
*   **Mecanismo**: En lugar de un KV-Cache que crece, usamos un **Acumulador Holográfico**.
*   **Codificación**: [VALIDADO v163e] Cada nuevo token se suma al acumulador tras aplicarle un `Roll(t, pos)`. Esto preserva el orden en un único vector.
*   **Atención**: El "Attention Head" no mira una lista de vectores, sino que interroga al holograma mediante resonancia de Walsh.
*   **Capacidad**: Compresión de contexto de al menos 4x-8x.

## 4. Estructura Profunda (Deep Stack)
*   **Jerarquía**: [VALIDADO v164b] Uso de **Residuos Espectrales**. La Capa 1 captura la intuición general y la Capa 2 refina el error o el detalle fino.
*   **Latencia**: Altamente eficiente. ~100-200ms por paso para 10k tokens en CPU.

## 5. Flujo de Inferencia
1.  **Token** $\to$ **Walsh Domain**.
2.  **Context Recall**: Resonar con el Holograma para recuperar el "estado mental" previo.
3.  **Knowledge FFN**: Consultar a los 131k expertos (nacidos por sorpresa o pre-entrenados).
4.  **Hologram Update**: Sumar el nuevo estado al acumulador con un desplazamiento circular.
5.  **Output**: Proyectar a logits de vocabulario.

## Hallazgos Integrados
- [x] **V163d**: Escalado a 131k expertos con latencia mínima.
- [x] **V163e**: Multiplexado espaciotemporal (memoria de orden).
- [x] **V164b**: Viabilidad de stacks profundos con residuos.
- [x] **V167b**: Neurogénesis dirigida por sorpresa (aprendizaje orgánico).

## Pendiente
- [ ] **Sparse Training (SWO)**: Refinar el aprendizaje de expertos existentes sin destruir el instinto inicial (híbrido v165c).
- [ ] **Inter-layer Holograms**: ¿Cómo fluye el contexto entre diferentes niveles de jerarquía?
