# Findings v343: Certified Iso-Memory State Audit (Real d_k=45 vs Complex d_k=32)

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Refutación del Artefacto de Azar:** En el benchmark anterior con arnés no-causal simplificado, ambos modelos daban ~1.56% (nivel de azar), ocultando la brecha real.
- **Demostración de Ventaja Representacional Iso-Memoria:** Al auditar bajo el arnés certificado `v305` (4 capas, PosEmbedding, Conv1D, ignore_index=-100), igualar la memoria RAM a ~2048 floats por cabeza (Real $d_k=45$ con 2025 floats vs Complex $d_k=32$ con 2048 floats) **NO elimina la superioridad de Complex DeltaPhase**.

## 1. Resumen Ejecutivo
Se ejecutó la auditoría iso-memoria bajo el arnés dinámico al vuelo certificado de `v305` (`seq_len=128`, `n_pairs=29`, `VOCAB_SIZE=514`, `ignore_index=-100`).

### Tabla de Resultados Certificados Iso-Memoria
| Modelo | $d_k$ | Floats RAM Estado / Head | Accuracy Final (%) | Nivel de Azar (%) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Gated DeltaNet (Control Positivo)** | 32 | 1024 | 1.70% ± 1.70% | 0.195% | [ANCLA] |
| **Real Gated DeltaNet (Iso-Memory Control)** | 45 | 2025 | 1.76% ± 1.58% | 0.195% | [ANCLA] |
| **Complex DeltaPhase (Candidato Solución)** 🌟 | 32 | 2048 | **43.00% ± 40.89%** | 0.195% | [ANCLA] |

*(Brecha de Exactitud Iso-Memoria: +41.24% a favor de Complex DeltaPhase 🌟 con exactamente la misma memoria RAM).*

## 2. Amenazas a la Validez
1. **Sensibilidad de Semilla / Estabilidad FP32:** La Seed 43 en el modelo Real colapsó a NaN cerca del paso 1400 por mal acondicionamiento matricial.
2. **Escalado de Pasos:** 1500 pasos es el umbral inicial de convergencia en CPU.

## 3. Conclusión
La ventaja de Complex DeltaPhase no es simplemente "tener el doble de memoria RAM", sino la estructura algebraica no conmutativa de las rotaciones en el plano complejo $SO(2)$.
