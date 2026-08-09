# Documento de Reconciliación y Refutación (Fase 0) — Auditoría del Corpus v300-v304

> 📜 **ESTATUS:** Documento de autocrítica rigurosa y reconciliación de datos. Este documento audita y refuta explícitamente las hipótesis sobre-interpretadas en el Informe Consolidado previa a la adopción de las nuevas reglas de reconciliación.

---

## 1. Falsificación de la Conclusión 1 mediante los Datos de v304

### La Hipótesis Previa (Conclusión 1):
> *"La fase en el círculo unitario complejo $S^1$ previene el cruce destructivo cuando claves y valores comparten embeddings… donde los controles reales colapsan."*

### La Evidencia Empírica de v304 (Tabla §2.4):

| Modelo | Parámetros | Val Loss | Val PPL ($e^{\text{Loss}}$) | Posición |
| :--- | :---: | :---: | :---: | :---: |
| **`ChunkwiseRealDeltaNetRectangular`** (Control Real) | 175,675 | **1.7811** | **5.94** | 1º (Ganador) |
| **`ChunkwiseComplexDeltaPhase`** | 144,331 | **1.7913** | **6.00** | 2º |
| **`CausalAttentionMHA`** (Softmax MHA) | 141,883 | **1.8506** | **6.36** | 3º |

### La Refutación:
En lenguaje natural autorregresivo a nivel de caracteres (*Tiny Shakespeare*), **todos los caracteres comparten por definición el mismo diccionario de embeddings**. Si las representaciones reales sufrieran un "colapso representacional intrínseco" por compartir vocabulario entre claves y valores, `ChunkwiseRealDeltaNetRectangular` no podría ser el modelo ganador del experimento (Val Loss 1.7811, PPL 5.94).

Un modelo no puede sufrir un colapso representacional invencible en una tarea sintética y al mismo tiempo ser el modelo con mejor capacidad de modelado en lenguaje real.  
**Conclusión:** La hipótesis de que la fase compleja es conceptualmente necesaria para evitar la interferencia de vocabulario compartido queda **falsificada**.

---

## 2. Diagnóstico del Bug del Harness Sintético (MQAR)

El contraste directo entre los resultados sintéticos y los de lenguaje natural actúa como el instrumento de debugging definitivo:

- `RealRectangular` en MQAR sintético fácil ($v300 / v302$): **0.90%** (azar).
- `RealRectangular` en lenguaje natural (*Tiny Shakespeare*, $v304$): **1.7811 Val Loss / 5.94 PPL** (Mejor de la serie).

### Diagnóstico Técnico:
Dado que el código de la capa `ChunkwiseRealDeltaNetRectangularBlock` es exactamente idéntico en ambos experimentos, **el modelo funciona correctamente**.  
El colapso a 0.90% en la tarea sintética no es una propiedad matemática de la representación real, sino un **bug en el harness sintético** (generador de secuencias MQAR, enmascaramiento de tokens `compute_kv_mask` o codificación posicional), el cual también provoca que la atención Softmax `CausalAttentionMHA` caiga al azar (0.21%) en contextos de $L > 500$.

---

## 3. Re-evaluación de v303 (30% Overwrite): Caída Severa de Rendimiento

### Los Datos Reales de v303:

| Modelo | 0% Overwrite | 30% Overwrite (ow30_k32) | Caída |
| :--- | :---: | :---: | :---: |
| **ChunkwiseComplexDeltaPhase** | 99.61% | **8.40%** | **-91.21 pp** |
| **ChunkwiseRealDeltaNetRectangular** | 0.90% (Bug Harness) | 0.54% | N/A |

### Re-interpretación Honesta:
Una caída de precisión del **99.61% al 8.40%** (un desplome de 91 puntos porcentuales) es una **incapacidad para resolver la sobreescritura de memoria** bajo la Delta Rule en 20 épocas.

Presentar 8.40% como "inicio de aprendizaje" comparándolo contra un baseline real averiado a 0.90% fue un error metodológico. La conclusión honesta es:  
*`ComplexDeltaPhase` no logra mantener la integridad de la memoria $M$ bajo sobreescritura activa de claves en el régimen probado.*

---

## 4. Corrección de Reglas de Anotación y Errores Metodológicos

1. **Eliminación del Resaltado Automático de Marca (🌟):**  
   Se elimina el uso de marcadores decorativos fijados a la identidad del modelo. Los marcadores de celda en futuros documentos responderán estrictamente a los óptimos de la columna/tabla (mínimo de Loss / máximo de Accuracy).
2. **Restauración de Marcas de Cautela y Anomalía (⚠️):**  
   Todas las tablas consolidadas mantendrán obligatoriamente las marcas ⚠️ cuando el techo teórico (Softmax MHA) o un control principal colapse, evitando la evaporación de advertencias en el pipeline de documentación.
3. **Control Iso-Parámetros en Lenguaje Natural:**  
   Se reconoce que en $v304$, `RealRectangular` contaba con un $21.7\%$ más de parámetros (175k vs 144k). Para aislar completamente el efecto, el siguiente barrido incluirá una variante real iso-parámetro exacta.

---

## 5. Implementación Permanente de la Regla de Reconciliación

A partir de este documento, se establece la **Regla de Reconciliación Obligatoria**:
> **Todo nuevo documento o informe debe comenzar por una sección titulada:**  
> *"Qué conclusión previa modifica o invalida este experimento"*  
> **No se permitirá publicar ningún informe sin auditar y declarar explícitamente los datos que contradigan hipótesis anteriores.**
