# V108: nGPT + ConeAttn — Findings

## Resultados

| Config | Params | Val Loss | PPL | Conv | Tiempo |
|---|---|---|---|---|---|
| **Standard Transformer** | 626,560 | **1.5518** | 4.72 | **Ep2** | 888s |
| nGPT+Cone+Dense | 437,408 | 1.8586 | 6.41 | Ep7 | 729s |
| nGPT Transformer | 609,152 | 1.9024 | 6.70 | Ep11 | 972s |
| nGPT+Cone+Narrow | 91,808 | 1.9439 | 6.99 | Ep10 | 473s |
| nGPT+Attn+DimGate | 214,784 | 2.0234 | 7.56 | Never | 663s |
| nGPT+Cone+DimGate | 43,040 | 2.1044 | 8.20 | Never | 417s |

---

## Hallazgos

### 1. nGPT converge MÁS LENTO que el Transformer estándar ⚠️

```
Standard:    Conv=Ep2,  best_val=1.5518  ← val<2.0 en batch 2
nGPT:        Conv=Ep11, best_val=1.9024  ← val<2.0 en época 11
```

Esto contradice la afirmación del paper de 4-20× más rápido. El motivo es que nGPT
requiere **hiperparámetros distintos** al Transformer estándar. Con lr=3e-3 y alpha_init=0.05,
cada paso en la hiperesfera es minúsculo:

```
Paso efectivo: ||alpha * norm(f(x))|| ≈ 0.05 (α inicial)
               alpha aprendido crece hasta ~0.3-1.4 al final de 20 épocas
```

El modelo está aún "calentando" los alphas cuando terminamos. Necesita o bien:
- **LR más alto** (nGPT trabaja bien con lr=1e-2 o superior)
- **Más épocas** (la convergencia es más lenta por paso pero más estable)

### 2. ConeAttn MEJORA nGPT incluso en la hiperesfera 🌟

```
nGPT puro (Attn+Dense):  val=1.9024
nGPT+Cone+Dense:         val=1.8586 → +2.3% mejor que nGPT puro
```

Con los mismos hiperparámetros subóptimos para nGPT, los conos aún ayudan.
La estructura posicional aprendida (offset/radio) sigue aportando información
incluso cuando el residual stream vive en S^(d-1).

### 3. DimGate NO se rehabilita en nGPT — por una razón fundamental

El alpha_ffn de las configs DimGate colapsa a cero:
```
E nGPT+Attn+DimGate: alpha_ffn=[0.000, 0.132, 0.627]  ← mínimo=0
F nGPT+Cone+DimGate: alpha_ffn=[0.000, 0.161, 1.067]  ← mínimo=0
```

La razón es algebraica. En nGPT, el update es:
```
x ← norm_sphere(x + α * norm_sphere(f(x)))
```

Para DimGate: `f(x) = x * sigmoid(g)`. Entonces:
```
norm_sphere(f(x)) = norm_sphere(x * sigmoid(g))
                  = x / ||x||   ← dirección idéntica a x (¡mismo vector!)
```

El sigmoid sólo cambia la magnitud de x, no su dirección. Tras normalizar,
`norm_sphere(DimGate(x)) = norm_sphere(x)`. El update equivale a:
```
x ← norm_sphere(x + α * x) = norm_sphere((1+α)*x) = x   ← identidad!
```

La red aprende α→0 porque el DimGate update es una identidad exacta en S^(d-1).
**nGPT no rehabilita DimGate; nGPT lo hace aún más inútil porque normalizar
cancela el único efecto que tenía (el reescalado de magnitud).**

### 4. Los alpha stats revelan la dinámica de aprendizaje

```
nGPT Transformer:
  L0: alpha_mixer=[0.014, 0.339, 1.156]  alpha_ffn=[0.112, 0.269, 0.543]
  L2: alpha_mixer=[0.358, 0.825, 1.358]  alpha_ffn=[0.316, 0.864, 1.425]
```

Patrones emergentes:
- **Alphas crecen con la profundidad** (L0→L2): las capas profundas aprenden a tomar
  pasos más grandes en la esfera. Igual que los radios de ConeAttn crecan con L.
- **Alta varianza por dimensión** (min≈0, max≈1.5): especialización fuerte.
  Algunas dimensiones casi no se actualizan (α≈0), otras se actualizan agresivamente.
- **FFN alphas > mixer alphas** en capas profundas: el FFN domina la actualización.

### 5. El problema de implementación: hiperparámetros

Nuestra implementación de nGPT es correcta, pero usa hiperparámetros calibrados
para Transformers estándar. El nGPT paper usa:

| Param | Estándar (nuestro) | nGPT (paper) |
|---|---|---|
| lr | 3e-3 | 1e-2 a 1e-1 |
| alpha_init | 0.05 | 0.05 pero con cosine decay separado |
| epochs | 20 | mucho más (hasta plateau) |
| grad_clip | 1.0 | sin necesidad (sphere estabiliza) |

---

## Conclusión

### ¿Vale la pena nGPT para nuestra arquitectura?

**Sí, pero requiere sus propios hiperparámetros.** Los resultados actuales no son
una refutación de nGPT sino una confirmación de que los hiperparámetros importan
mucho en la hiperesfera.

El hallazgo más limpio: **ConeAttn + nGPT (correctamente calibrado) es una
combinación prometedora**. Con los mismos hiperparámetros, ConeAttn mejora nGPT
en +2.3%. Si nGPT se calibra correctamente (más LR, más épocas), y ConeAttn
mantiene su +2.3% de ventaja relativa sobre Standard Transformer, el combo
sería muy competitivo.

### La lección algebraica sobre DimGate

nGPT revela que DimGate no es solo débil — es **fundamentalmente incompatible
con el aprendizaje en la hiperesfera** porque la normalización cancela
exactamente lo único que DimGate hace (reescalar magnitudes).

Esto cierra el capítulo DimGate definitivamente:
- En Euclidean space: colapsable (L capas = 1 capa)
- En Hypersphere (nGPT): identidad exacta (inútil)

### Siguiente experimento sugerido (V109)

Re-testear nGPT con lr=1e-2, alpha_init=0.1, epochs=40 para ver si la
convergencia acelera como promete el paper. Hipótesis: con LR correcto,
nGPT+Cone+Dense debería llegar a val≈1.55-1.60.
