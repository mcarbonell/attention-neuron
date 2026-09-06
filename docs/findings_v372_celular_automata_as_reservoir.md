# v372 — CA como Reservorio (Reservoir Computing)

**Fecha:** 2026-08-28
**Lab:** `attention-neuron/`
**Script:** `scratch/prototype_v372_ca_reservoir.py`
**Hipótesis a probar:** ¿es útil una capa CA como extractor si dejamos la regla
**congelada** (no entrenada) y solo entrenamos un readout encima? (Estilo
*Reservoir Computing*: la dinámica recurrente es un "reservorio" barato y solo se
entrena la salida.)

## Diseño
- Reglas **congeladas** (sin gradiente): `RandomCARule` (MLP 3×3 random), `Life`
  (suave), `PolyBlend` (Life+Seeds+HighLife con α fijo). El CA itera `T=20`
  pasos sobre el input binario.
- **Lo único entrenado**: el readout. Dos variantes:
  - *Lineal* sobre estadísticas globales de los últimos `K=10` pasos (mean/std/max).
  - *Convolucional* pequeña sobre la grilla final.
- Baseline `identity`: mismos readouts pero sobre la imagen cruda (sin CA).

## Resultados (MNIST, 3000 train / 1000 test)
| Configuración | linear readout | conv readout |
|---|---|---|
| **identity (raw image)** | 0.186 | **0.613** |
| random_CA (seed0) | 0.157 | 0.174 |
| random_CA (seed1) | 0.126 | 0.126 |
| Life (smooth) | 0.108 | 0.204 |
| polyblend (Life+Seeds+HighLife) | 0.118 | 0.235 |

## Interpretación (hallazgo central)
**Un CA no aprendido, como reservorio, DESTRUYE la señal de clasificación.**
- El random_CA deja el conv-readout en 0.126: la dinámica aleatoria **borra** el
  patrón espacial del dígito. El mapa random (tanh/sigmoid) lleva la grilla a
  punto fijo o caos, perdiendo el input (memoria que se desvanece mal ajustada).
- Life/polyblend conservan algo más (0.20–0.24) pero siguen muy por debajo del
  baseline `identity+conv = 0.613`.
- Conclusión: la estructura CA *por sí sola* no aporta para clasificar imágenes
  estáticas. Lo que hizo funcionar v371 (0.64) fue **aprender la regla** para
  preservar la información discriminativa — no el CA en sí.

## Reconciliación con la literatura
El éxito de los CA-reservoir en papers (Yilmaz, Gilpin, etc.) ocurre en regímenes
específicos: reglas en el **edge of chaos** (λ de Langton bien tuneado), input
*sostenido* en el tiempo, y tareas donde el CA es natural (morfogénesis,
dinámica temporal). Clasificar imágenes estáticas MNIST no es su fuerte, y un
random thresholded CA típicamente borra info. Nuestro resultado es coherente:
sin el ajuste fino de λ o del readout temporal, el reservorio no ayuda.

## Qué aprendemos para el lab
1. **El aprendizaje de la regla es el ingrediente clave**, no la topología local.
   v371 (regla aprendida) ≫ v372 (regla congelada).
2. El CA brilla donde la CNN no: **auto-organización / morfogénesis / self-repair**
   (Mordvintsev et al. 2020), no como clasificador feedforward estático.
3. Un reservorio CA útil exigiría: regla en edge-of-chaos, input sostenido, y
   readout entrenado sobre *toda la trayectoria* (no solo el último paso).

## Siguiente paso sugerido (v373)
Pivotear a la demo donde el CA es genuinamente superior: **Neural CA de
morfogénesis / auto-reparación** — crecer un dígito MNIST y borrarle un pedazo
para verlo reconstruirse. Ahí la recurrencia y la regla local son el punto fuerte,
no un obstáculo.

## Cómo reproducir
```powershell
cd attention-neuron/scratch
python prototype_v372_ca_reservoir.py
# escribe attention-neuron/docs/v372_findings.json
```
*(El warning de "buffer not writable" al cargar IDX es inofensivo.)*
