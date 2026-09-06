# v371 — Neurona-Autómata Celular Polimórfica con Regla APRENDIDA (Neural CA)

**Fecha:** 2026-08-28
**Lab:** `attention-neuron/`
**Script:** `scratch/prototype_v371_nca_polymorphic.py`
**Basado en:** v370 (reglas fijas outer-totalísticas + lerp de alpha).

## Qué cambió respecto a v370
1. **Regla aprendida (no fija):** cada regla es un MLP `9 → 16 → 1` que toma la
   vecindad 3×3 completa (los 9 valores, no solo centro+suma) y produce un
   **delta de estado** (estilo Neural Cellular Automata de DeepMind).
2. **Actualización residual + init ~0:** `grid_next = clamp(grid + Σ α_k·rule_k, 0, 1)`.
   El MLP se inicializa con el último layer en cero ⇒ el CA arranca como
   **identidad** y preserva la señal de entrada. (Esto fue clave; ver abajo.)
3. **Readout local convolucional** en vez de pooling global (sugerencia v370-1).
4. Sigue habiendo `K=3` reglas mezcladas por `α = softmax(W)` (polimorfismo) y
   auto-recurrencia de `iters=2`.

## Resultados
| Experimento | v370 (regla fija) | v371 (regla aprendida) |
|---|---|---|
| MNIST val_acc | 0.13 (chance) | **0.54–0.64** |
| Parámetros | 496 | 1904 |

- **Hallazgo crítico — el init:** sin la inicialización residual ~0, el CA
  **colapsaba** y el clasificador se quedaba en loss 2.30 (random). Con identidad
  inicial, el head aprende del imagen real y luego la regla lo refina. Esto es
  generalizable a cualquier arquitectura CA: la dinámica iterada necesita arrancar
  cercana a identidad o se destruye la información.
- **Recuperación de regla (prueba directa):** forzar al CA a reproducir
  `Life` 1 paso → `α = [0.94, 0.027, 0.037]`. El modelo **concentra casi todo el
  peso en una sola regla** para emular Life. Demuestra que el lerp polimórfico
  *aprende a seleccionar/aproximar* una regla objetivo. `final_mse ≈ 0.16`
  (ruido de discretización + tanh; no es cero, pero la especialización es clara).

## Interpretación
- **Reglas aprendidas ≫ reglas fijas** para tareas de gradiente: confirma la tesis
  de v370 de que "aprender la regla navega solo a la banda útil" (edge of chaos),
  sin tener que enumerar las 512.
- El polimorfismo de `α` es funcional pero, en la práctica, el optimizador
  **colapsa a 1 regla** cuando una basta. El valor del polimorfismo aparece cuando
  ninguna regla aislada basta (tareas multi-régimen) — sugeriría `α` *por capa* o
  *por paso de recurrencia* en vez de global.

## Limitaciones observadas
- Una regla fija mezclada no reproduce bien **Life a 2+ pasos** (mse sube): aplicar
  la misma regla dos veces ≠ dos reglas distintas. Apunta a `α` dependiente del
  paso de recurrencia.
- MNIST aún lejos de una CNN normal (~0.99); aquí la meta era probar el mecanismo,
  no competir en accuracy.

## Siguientes pasos sugeridos
1. **`α` por paso de recurrencia** (gating temporal) para emular dinámicas multietapa.
2. **Regla de patrón completo de 9 bits** (tabla 512 con soft-indexing) para reglas
   no outer-totalísticas.
3. **Gating duro/combinatorio** (top-k reglas) en vez de softmax suave.
4. Cruzar con las bases espectrales del lab (v322+) para un CA híbrido
   espectral+celular.

## Cómo reproducir
```powershell
cd attention-neuron/scratch
python prototype_v371_nca_polymorphic.py
# escribe attention-neuron/docs/v371_findings.json
```
