# Reglas estables del Repositorio

## Filosofía y Visión
**EL NORTE DEL PROYECTO:** El objetivo sagrado no es el Accuracy por sí mismo. Buscamos:
1. **Eficiencia Paramétrica y Algorítmica:** Hacer más con menos. Un sistema 100x más comprimido es superior a uno con +1% de precisión pero pesado.
2. **Originalidad y Ruptura de Dogmas:** Cuestionar las bases de la IA actual (ej. backpropagation, capas densas) y proponer alternativas espectrales, holográficas u orgánicas.
3. **Elegancia sobre Fuerza Bruta:** Valoramos la belleza del algoritmo, la interpretabilidad y la comprensión profunda de los datos por encima de los resultados obtenidos mediante cómputo masivo.
4. **Exploración amplia antes que explotación profunda:** El objetivo primario no es cerrar líneas de investigación con validación exhaustiva, sino cubrir el espacio de hipótesis con puntos dispersos de bajo coste que construyan una visión global. La profundidad y el rigor máximo se reservan para los hallazgos que se promuevan a ANCLA (ver Niveles de Rigor).

## Modificación de código existente
**REGLA DE ORO (No modificar experimentos previos):** Bajo NINGÚN CONCEPTO debes modificar un archivo de código (`.py`) que ya exista en `scratch/` y funcione.
- Todas las iteraciones y mejoras algorítmicas deben realizarse creando NUEVOS archivos.
- Es preferible y totalmente aceptable duplicar código entre archivos si esto evita tocar implementaciones pasadas que ya son estables y sirven de referencia.

## Protocolo Obligatorio Antes de Declarar un Resultado Negativo como Definitivo

Antes de escribir cualquier conclusión que cierre, descarte o declare "no funciona"
una hipótesis, el agente debe descartar EXPLÍCITAMENTE, una por una, las siguientes
causas alternativas — no basta con proponer una explicación narrativa plausible
que las sustituya:

1. **¿Hay un bug de implementación?** Verificar con un test unitario mínimo
   (ej. forward pass sobre un caso trivial con solución conocida a mano).
2. **¿El baseline de comparación está bien ajustado?** (LR, warmup, épocas suficientes
   — no solo copiado de otro experimento con setup distinto).
3. **¿Falta algún paso de preprocesamiento que otro experimento de la misma familia
   sí incluye?** (ej. normalización, reordenamiento, escalado) — revisar
   explícitamente el `master_ledger.jsonl` en busca de experimentos de la misma
   familia con setup ligeramente distinto antes de concluir.
4. **¿El fallo es sensible a un hiperparámetro no barrido?** (ej. el ratio de
   compresión, el rango k, el learning rate) — al menos 3 valores distintos
   antes de declarar fallo general.
5. **¿La métrica de evaluación tiene suficiente muestra?** (ver regla de SE) —
   un colapso puede ser ruido de medición, no fallo real.

Solo si las 5 causas anteriores se descartan CON UN EXPERIMENTO CONCRETO (no con
una explicación teórica sin testear), el hallazgo puede etiquetarse [ANCLA-NEGATIVO]
("esto no funciona, confirmado") en vez de [CIERRE-PREMATURO-SOSPECHA]
(default hasta que se complete el checklist).


## Niveles de Rigor (obligatorio clasificar cada experimento)

### Nivel 1 — Sondeo Exploratorio (por defecto, la mayoría de experimentos)
- 1 semilla es aceptable. Dataset de evaluación pequeño permitido.
- Logging mínimo: config + resultado crudo en JSON + línea en el Master Ledger (ver abajo).
- Etiqueta obligatoria de cada hallazgo: [SEÑAL] o [RUIDO-SOSPECHA].
- **Prohibido** citar un resultado de Nivel 1 como "supera al baseline/float32" o como evidencia fuerte. Solo se reporta como punto exploratorio a seguir o descartar.

### Nivel 2 — Hallazgo Candidato a [ANCLA]
Se promueve un experimento a Nivel 2 cuando:
- (a) el resultado de Nivel 1 sugiere algo que cambiaría dónde inviertes las próximas semanas, o
- (b) vas a citarlo fuera del repo (post público, doc de venta, conversación con un lab/reclutador/inversor).

Requisitos obligatorios en Nivel 2:
- Mínimo **5 semillas** por configuración de hiperparámetros.
- Dataset de evaluación grande (evitar datasets <1000 secuencias independientes para métricas tipo perplejidad).
- Cálculo de error estándar (SE) según el procedimiento de la sección de Logging.
- Todas las métricas de Diagnóstico y Reproducibilidad completas (ver Logging).
- Solo entonces puede pasar de [SEÑAL] a [ANCLA].

## Metodología de Investigación
Para cada nueva iteración o experimento algorítmico:
1. **Nueva Versión**: Crear un nuevo archivo en `scratch/` (p.ej. `prototype_vX.py`).
2. **Clasificar Nivel de Rigor**: Decidir explícitamente si es Nivel 1 o Nivel 2 antes de ejecutar.
3. **Experimento**: Ejecutar las pruebas y recoger métricas según el nivel correspondiente.
4. **Documentación**: Crear `findings_vX.md` en `docs/` siguiendo el formato obligatorio (ver Reglas de Reporte).
5. **Ledger**: Añadir la línea correspondiente a `results/master_ledger.jsonl`.
6. **Commit**: Realizar un commit con código, documentación y ledger antes de pasar a la siguiente fase.
7. **Iteración**: Proponer y ejecutar el siguiente experimento basado en los hallazgos previos.

## Gestión de Ejecución y Cuota (Consumo de Tokens)
**REGLA DE ORO (No lanzar procesos):**
- **Ejecución por defecto:** Las ejecuciones de benchmarks, entrenamientos y pruebas largas las debe realizar el **USER** directamente en su terminal. El agente no debe lanzarlas por iniciativa propia. Puede que ya haya un proceso pesado ejecutándose.
- **Ejecución bajo demanda:** Si el USER pide explícitamente al agente que lance un script, el agente **NO** debe lanzarlo en modo background si esto implica permanecer activo muestreando la salida (consume tokens rápidamente).
- **Espera de finalización:** Si el agente lanza un script, debe hacerlo de forma que el control no vuelva a él hasta que el script termine, o debe entrar en pausa hasta recibir la señal de finalización — evitando el "sampling" continuo de logs en segundo plano.
- **Regla de Oro (Primero el Candidato):** En scripts de benchmark, el algoritmo/modelo que se está probando o iterando DEBE ejecutarse siempre en primer lugar. Los baselines y referencias se ejecutan después, para detectar fallos o bugs instantáneamente.

**REGLA DE ORO (Vectorización o Muerte):** En arquitecturas espectrales (Hadamard, FFT) o de gating masivo, queda terminantemente PROHIBIDO usar bucles `for` de Python para operaciones tensoriales en el `forward`.
- Toda operación de mezcla o proyección debe realizarse mediante multiplicaciones de matrices (`@`) o funciones nativas de PyTorch.
- Si una operación (como Hadamard) es recursiva, debe precomputarse en una matriz buffer durante el `__init__`.
- El fallo en seguir esta regla puede causar que una época pase de 30 segundos a 1 hora, inutilizando el experimento.

**Hardware:**
- GPU: `C:/Users/mrcm_/Local/proj/ajedrez/neural-tablebases/venv_gpu/Scripts/python.exe` (v3.12, Torch DirectML).
- Redes pequeñas: normalmente más rápido en CPU con `python.exe` v3.13.

## Normas de Logging y Resultados (Sistema de Métricas)

### 1. Desempeño y Coste (Efficiency)
- `final_objective`: Valor final alcanzado (Loss, Accuracy, etc.)
- `total_evaluations`: Número total de llamadas a la función f(x)
- `wall_clock_time`: Tiempo real total transcurrido
- `function_evaluation_time`: Tiempo neto gastado en forward passes (f(x))
- `internal_overhead_time`: Tiempo neto gastado por la lógica del optimizador (EMA, particionado, Adam). Calculado como `(WallClock - EvalTime)`.
- `PEI` (Parametric Efficiency Index): `Accuracy / log10(TotalParams + 1)`. Nota: esta métrica aplana mucho las diferencias entre escalas muy distintas (ej. 1K vs 100M params). Para claims de tipo "N veces menos parámetros", complementar con `params_a_iso_loss`: parámetros que necesitaría el baseline denso (interpolando su curva de escalado) para igualar la loss obtenida.

### 2. Estabilidad y Rigor (Robustness) — obligatorio en Nivel 2, opcional en Nivel 1
- `num_seeds`: Mínimo 5 semillas por configuración en Nivel 2.
- `std_objective`: Desviación estándar entre semillas.
- `convergence_speed`: Número de evaluaciones necesarias para alcanzar el 90% del objetivo final.

### 3. Señal del Optimizador (Diagnostics)
- `snr_correlation`: Correlación de Pearson entre el gradiente ruidoso del paso actual y el acumulado (EMA).
- `gradient_sparsity`: Porcentaje de parámetros con gradiente acumulado nulo o despreciable.
- `step_efficiency`: Mejora media del objetivo por cada evaluación de función.

### 4. Entorno de Ejecución (Reproducibility)
- `commit_hash`: Hash exacto del código que generó el resultado.
- `hardware_info`: CPU, GPU (si aplica) y memoria disponible.
- `full_config`: Copia completa del JSON de hiperparámetros.

### 5. Contrato de trazabilidad del log — obligatorio para todo script de experimento

Todo script nuevo de entrenamiento, benchmark o evaluación debe implementar las siguientes reglas. No basta con que la información exista en el código: debe quedar impresa en el log y guardada en el JSON crudo.

1. **Marca temporal y salida sin búfer:** toda línea emitida por el script debe comenzar con una marca de tiempo (`[+HH:MM:SS.ss]` o `[HH:MM:SS]`). Todo `print()` debe incluir `flush=True` (o ejecutar el script con `python -u`) para garantizar la transmisión en vivo por consola sin buffering.
2. **Metadatos de ejecución y explicación al inicio:** cabecera clara con ID de experimento, hipótesis a probar, fecha UTC, comando/argumentos, commit hash, versión de Python y PyTorch, dispositivo, CPU/GPU, plataforma, estado de determinismo e inventario completo de arquitectura (parámetros totales/entrenables, dimensiones $d_{\text{model}}, n_{\text{heads}}, d_k$, capas y desglose por módulo).
3. **Configuración reproducible completa:** imprimir y persistir el JSON completo de hiperparámetros, semillas, rutas/versiones de datos, tamaños de split, presupuesto de tokens/pasos y criterios de selección de checkpoint.
4. **Descripción arquitectónica por capas en cabecera:** antes de entrenar cada configuración, registrar un inventario ordenado de capas/bloques, dimensiones, tipo de mezclador de secuencia, tipo de FFN, parámetros entrenables por componente y total. Los nombres de marketing no sustituyen esta descripción.
5. **Feedback y monitoreo continuo en tiempo real:** registrar periódicamente paso/época actual, porcentaje de avance, pérdida reciente, métricas de evaluación retenidas, velocidad en pasos por segundo (`st/s`) y **estimación de tiempo restante (ETA tanto para el modelo en curso como para la suite total)**. El JSON crudo debe conservar el historial completo.
6. **Evaluación separada:** imprimir en cada época la métrica de train y la de validación con número de secuencias evaluadas; el test sólo se ejecuta sobre el checkpoint elegido por validación. Nunca presentar una métrica de entrenamiento como validación.
7. **Persistencia final:** guardar el log estructurado completo, la configuración, el inventario de arquitectura, el historial por época/paso, métricas por semilla y el resumen en `results/raw/`. El ledger sólo se añade tras finalizar correctamente la ejecución.

### Cálculo de Error Estándar (SE) — obligatorio en Nivel 2
Para métricas de perplejidad/loss, **NO calcular SE token-a-token** (la autocorrelación dentro de una secuencia invalida el supuesto i.i.d. y subestima el error real).
- Calcular el loss medio **por secuencia** (no por token).
- `SE = std(loss_por_secuencia) / sqrt(n_secuencias)`.
- Mínimo 30 secuencias independientes para que el SE sea mínimamente confiable. Si hay menos, el resultado se etiqueta automáticamente [RUIDO-SOSPECHA], sin excepción, y no puede promoverse a [ANCLA].

### Master Ledger (obligatorio para TODO experimento, incluso Nivel 1)
Cada experimento añade una línea a `results/master_ledger.jsonl`:
```json
{
  "experiment_id": "vXXX",
  "fecha": "YYYY-MM-DD",
  "familia": " geométrico | espectral | optimizador | ... ",
  "dataset": "nombre y tamaño (n_secuencias / n_tokens)",
  "n_eval": 0,
  "metric_name": "ppl | acc | loss",
  "value": 0.0,
  "SE": null,
  "params": 0,
  "nivel_rigor": 1,
  "etiqueta": "ANCLA | SEÑAL | RUIDO-SOSPECHA"
}
```
Este ledger es lo que permite construir la visión global del espacio de hipótesis (ej. gráfico de PEI vs. params coloreado por familia) sin depender de releer cada `findings_vX.md` manualmente.

### Almacenamiento de Resultados
- Resultados crudos: `results/raw/*.json`.
- Resúmenes estadísticos: `results/summary/`.
- Gráficas comparativas: `results/figures/`.
- **REGLA DE ORO DEL LOGGING:** Ninguna afirmación de mejora es válida si no viene acompañada de un archivo JSON que demuestre que `internal_overhead_time` no anula el ahorro en `total_evaluations`.
- **REGLA DE SUPERVIVENCIA (Fast Feedback):** Todo script de entrenamiento (`.py`) DEBE imprimir información de progreso (Loss, etc.) en los **primeros 5 batches** de la Época 1, para confirmar instantáneamente que la red compila, el grafo fluye y el proceso no se ha colgado.
10. **Regla de Reconciliación Obligatoria entre Documentos:**
    Todo nuevo documento (`findings_vX.md` o informe consolidado) DEBE abrir obligatoriamente con una sección titulada:
    `## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento`
    - Es obligatorio declarar y auditar explícitamente los datos que contradigan o refuten hipótesis de experimentos previos (ej. contrastando resultados sintéticos vs texto real).
    - Queda terminantemente prohibido generar documentos en silos donde una tabla invalide verbalmente la conclusión del mismo archivo o de versiones anteriores sin señalar la refutación explícita.
11. **Rigor en Anotaciones y Marcadores (🌟 / ⚠️):**
    - El símbolo de éxito 🌟 se asigna **única y exclusivamente al modelo con el mejor valor numérico real de la columna/celda** (mínima Loss/PPL o máxima Accuracy). Queda prohibido aplicar marcadores por identidad del modelo ("nuestra arquitectura").
    - Las marcas de anomalía ⚠️ o colapso de baselines NO deben borrarse durante las fases de resumen o consolidación. Deben conservarse obligatoriamente para preservar la auditabilidad del corpus.
12. **Protocolo de Reconciliación Sintética vs Real:**
    - Si un modelo rinde de forma deficiente en una tarea sintética (ej. 0.90% en MQAR) pero es el mejor modelo en lenguaje natural real (ej. 1.7811 Val Loss en texto), se debe declarar inmediatamente la existencia de un **bug o artefacto en el harness sintético**, en lugar de inventar explicaciones representacionales sin testear.

## Reglas de Reporte (obligatorias, sin excepción)

1. **Reporta siempre el número.** Prohibido "explosión" o "colapso" sin valor; usa ">X (saturado)".
2. **Toda afirmación en conclusiones cita la celda concreta de la tabla que la soporta.**
3. **Ninguna "mejora" es válida si |Δ| en nats < 2× el SE del eval.** Calcula el SE (ver procedimiento arriba) y repórtalo explícitamente.
4. **Palabras prohibidas** (usar en su lugar: "sugiere", "se observa", "consistente con", "no distinguible del ruido"):
   demuestra, destruye, teorema, óptimo, extraordinario, revolucionario, hito, nacimiento de una [nueva] arquitectura, sin precedentes, asombroso, la Santísima Trinidad, "supera al baseline" (sin el cálculo del punto 3).
5. **Toda curva no monótona se marca como anomalía** con causa propuesta (bug, ruido, régimen). Nunca se ignora ni se omite de la tabla.
6. **Sección final obligatoria: "Amenazas a la Validez"** — las 3 objeciones más fuertes contra tus propias conclusiones y el experimento que las dirimiría.
7. **Etiqueta cada hallazgo:** [ANCLA] verificado (Nivel 2 completo) / [SEÑAL] sin confirmar (Nivel 1) / [RUIDO-SOSPECHA] probablemente artefacto.
8. **Reporta primero el control/baseline** y compáralo con la referencia conocida si existe (de un experimento anterior). Si difiere, **detén todo** y repórtalo como fallo de harness, no como resultado nuevo.
9. **Coincidencia de alcance al citar baselines anteriores:** al usar un resultado de un experimento previo como referencia/baseline, verifica explícitamente que coincide en capas afectadas, dimensión de la transformada, dataset y tamaño de evaluación. Si algo difiere, **re-ejecuta el baseline dentro del experimento actual** — nunca cites un número de otro contexto sin re-verificar la equivalencia de setup.
