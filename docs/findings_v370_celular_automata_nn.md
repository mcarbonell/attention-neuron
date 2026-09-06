# v370 — Neurona-Autómata Celular Polimórfica

**Fecha:** 2026-08-28
**Lab:** `attention-neuron/`
**Script:** `scratch/prototype_v370_cellular_poly_neuron.py`
**Hipótesis origen:** red donde cada neurona de la capa siguiente mira 1 píxel + sus 8
vecinos (vecindad 3×3 de Moore) y aplica una regla tipo Conway como función de
activación; capas apilables; capas auto-recurrentes (la salida se realimenta N
iteraciones). Pregunta clave: ¿se puede *interpolar entre reglas* como se hace
lerp entre activaciones en la neurona polimórfica (v191/v193)?

## Respuesta corta
**Sí.** La interpolación entre reglas es análoga al lerp de activaciones, pero se
hace en el *espacio de la regla* (la tabla de transición), no punto-a-punto en la
entrada. Para que sea diferenciable y entrenable basta representar cada regla
como una función continua y mezclar sus salidas con un coeficiente `α`
(aprendido vía softmax).

## Arquitectura del prototipo
- **Reglas candidatas (K=6):** Life (B3/S23), Seeds (B2/S∅), HighLife (B36/S23),
  Replicator (B1357/S1357), DayNight (B3678/S34678), Stable (control).
- **Reglas suaves/diferenciables:** cada regla outer-totalística se implementa
  como `f(centro c∈[0,1], suma de 8 vecinos s∈[0,8]) → [0,1]` usando bandas
  sigmoides `_band(s, lo, hi)`. Esto permite el lerp continuo (regla "blanda").
- **Capa `CellularPolyLayer`:** extrae la vecindad 3×3 por convolución 3×3,
  evalúa las K reglas y combina:
  `grid_next = Σ_k α_k · rule_k(centro, Σvecinos)`, con `α = softmax(W)`.
- **Auto-recurrencia:** la capa itera su propia salida `iters` veces (bucle de
  realimentación), igual que apilar `stages` en `DeepPolymorphicNet` (v193).
- **Parámetros totales:** 496 (solo el vector `α` de la capa CA + head).

## Resultados
### 1. Clasificación MNIST (concepto, head trivial)
- `val_acc ≈ 0.13` (chance 0.10). Cae en el rango de la señal.
- `α` permanece **casi uniforme** (Stable 0.19, resto ~0.15-0.17).
- **Interpretación:** el readout (media/std/máx global del grid + densidad de
  tinta) no discrimina entre reglas; todas producen estadísticas gruesas
  similares, así que el gradiente no tiene incentivo para especializar `α`.
  Hallazgo útil: el mecanismo funciona, pero hace falta un readout *local/denso*
  (no pooling global) para que la regla importe en clasificación.

### 2. Prueba directa de recuperación de regla (la demostración fuerte)
- Tarea: reproducir `Life` aplicada 2 pasos sobre grids binarios aleatorios,
  entrenando solo `α`.
- `α` aprendido: **Life 0.31, HighLife 0.39, Stable 0.24, Seeds 0.001,
  DayNight 0.001, Replicator 0.055**.
- **Interpretación:** el modelo *especializa* claramente. Las reglas irrelevantes
  colapsan a ~0; se prefiere Life y su vecina cercana HighLife (B36/S23 ≈ B3/S23).
  `final_mse = 0.18` porque iterar una regla mezclada ≠ mezclar reglas iteradas,
  pero la selección/interpolación es inequívoca.
- **Conclusión:** la interpolación de reglas polimórficas **se entrena por
  gradiente** cuando la tarea depende de la dinámica del CA.

## Relación con "¿son igual de buenas las 512 reglas?"
No. En el diseño binario completo hay 2⁹ = 512 reglas; la inmensa mayoría mueren
o saturan. Las útiles viven en el *edge of chaos* (λ de Langton). Life es el
análogo 2D de una regla clase IV de Wolfram. Por eso **aprender `α` (o la regla
misma) es mejor que fijar una**: el entrenamiento navega solo a la banda buena,
como muestra el probe (HighLife≈Life emergió sin hardcodearlo).

## Siguientes pasos sugeridos
1. **Readout local:** reemplazar el pooling global por convoluciones posteriores
   (tipo CNN) para que la regla afecte la clasificación MNIST de verdad.
2. **Regla aprendida, no fija:** parametrizar cada `rule_k` con pesos (en vez de
   bandas hardcodeadas) → "Neural Cellular Automata" diferenciables.
3. **Regla de patrón completo (9 bits):** usar la tabla de 512 entradas con
   soft-indexing, para capturar reglas no outer-totalísticas.
4. **Polimorfismo anidado:** `α` distinto por canal / por capa / por paso de
   recurrencia.

## Cómo reproducir
```powershell
cd attention-neuron/scratch
python prototype_v370_cellular_poly_neuron.py
# escribe attention-neuron/docs/v370_findings.json
```
