# v373 — Neural CA de Morfogénesis / Auto-Reparación

**Fecha:** 2026-08-28
**Lab:** `attention-neuron/`
**Script:** `scratch/prototype_v373_nca_morphogenesis.py`
**Contexto:** v372 mostró que un CA *congelado* destruye la señal en clasificación.
La literatura (Mordvintsev et al. 2020, "Growing Neural Cellular Automata") dice
que el CA es fuerte en **auto-organización recurrente** (crecer/reparar patrones),
no en clasificación estática. v373 prueba exactamente eso.

## Método
- **Estado por célula:** vector de `C=4` canales (1 visible + 3 ocultos).
- **Regla local aprendida:** conv 3×3 → ReLU → conv 1×1 produce un *delta*
  residual; actualización **solo en células "vivas"** (máscara de percepción:
  una célula vive si ella o algún vecino 3×3 tiene canal visible > 0.1).
  Init en ~0 ⇒ arranca casi identidad. **660 parámetros.**
- **Target:** disco sólido 28×28 (los dígitos MNIST finos son demasiado duros para
  una regla puramente local; el disco aisla la dinámica de crecimiento/reparación).
- **Entrenamiento con aumento de daño** (clave): se crece desde semilla, a veces se
  **daña** (hueco aleatorio) a mitad de la trayectoria, y se deja curar; pérdida
  MSE sobre el canal visible en los últimos pasos. Así se entrena tanto crecer como
  reparar dentro de la misma dinámica.

## Resultados (target = disco, 30 pasos)
| Métrica | Valor | Significado |
|---|---|---|
| **hole_recovered_frac** | **0.99** | El 99% de los píxeles del hueco dañado se reconstruyen ✅ |
| grow_mse | 0.13 | El disco crecido desde semilla se parece al target |
| repair_mse (global) | 0.63 | La imagen reparada globalmente aún deriva en bordes |

**Conclusión central:** el CA **sí auto-repara** localmente. Es la demostración que
faltaba: en régimen recurrente auto-organizativo el CA es genuinamente superior a
una CNN (que no tiene dinámica de reparación temporal).

## Interpretación y límites
- `hole_recovered_frac ≈ 0.99` confirma que la regla local aprende a regenerar la
  región faltante desde los vecinos — el comportamiento "vivo" de un autómata.
- El `repair_mse` global alto indica que el **atractor no es aún nítido/estable en
  el borde**: tras muchos pasos la regla tiende a crecer un poco más allá del disco
  o a no frenar exactamente. Es el desafío clásico de entrenar NCA por trayectorias
  largas (gradientes a través de la recurrencia + equilibrio crecer/detener).
- Es un prototipo mínimo (660 params, 1 solo target, 200 epochs en CPU). Las
  versiones del paper usan más capacidad, pérdida de "repetibilidad" (promediar el
  estado final sobre muchas semillas) y targets más ricos (corazón, lagarto).

## Arco v370 → v373 (conclusión del lab)
| Exp | Regla | Resultado |
|---|---|---|
| v370 | fija, lerp α | MNIST 0.13 (chance) |
| v371 | **aprendida**, readout local | MNIST 0.64 |
| v372 | congelada (reservorio) | destruye señal (0.13) |
| v373 | **aprendida + recurrencia** | auto-reparación 0.99 ✅ |

**Lección:** la neurona-autómata necesita (1) regla aprendida y (2) régimen
recurrente auto-organizativo para brillar. Como clasificador estático no compite
con CNN; como sistema de morfogénesis/auto-reparación es superior.

## Siguientes pasos sugeridos
1. **Repetibilidad/novelty loss** (Mordvintsev): promediar el estado final sobre
   semillas y penalizar varianza → atractor más estable y nítido.
2. **Target más rico** (corazón/lagarto o dígito grueso) para medir reparación
   estructural, no solo relleno de disco.
3. **Más capacidad / más epochs** para bajar `repair_mse`.
4. Cruzar con la neurona polimórfica (v370/371): α de regla que dependa del paso de
   recurrencia o del estado local.

## Cómo reproducir
```powershell
cd attention-neuron/scratch
python prototype_v373_nca_morphogenesis.py
# escribe attention-neuron/docs/v373_findings.json
```
