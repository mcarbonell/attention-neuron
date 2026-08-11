# Estándar versionado de logging experimental

> Este documento es la copia versionada del contrato de trazabilidad incorporado en `GEMINI.md`. Se mantiene aquí porque `GEMINI.md` es una configuración local ignorada por Git.

Todo script nuevo de entrenamiento, benchmark o evaluación debe cumplir estas reglas:

1. Cada línea emitida comienza con una marca temporal relativa al lanzamiento: `[+HH:MM:SS.ss]`.
2. La cabecera registra ID, fecha UTC, argumentos, commit, versiones Python/PyTorch, hardware/dispositivo y determinismo.
3. Se imprime y persiste el JSON completo de configuración, semillas, datos/splits, presupuesto de pasos/tokens y criterio de checkpoint.
4. Antes de cada ejecución se describe la arquitectura por capas: orden, dimensiones, mezclador de secuencia, FFN y parámetros por componente y total.
5. Se registra cada época con train/valid loss, LR, norma media/final de gradiente, tokens, tiempos y checkpoint. El detalle por paso se activa con `--log-every N` sólo cuando hace falta depurar; el JSON conserva el historial completo por época.
6. Train, validación y test se distinguen explícitamente. El checkpoint se selecciona sólo por validación; el test se ejecuta una vez sobre ese checkpoint.
7. El JSON crudo conserva metadatos, inventario arquitectónico, historial, métricas por semilla y resumen; el ledger se escribe únicamente tras éxito.

Véase [GEMINI.md](../GEMINI.md) para las reglas completas de rigor, reconciliación y reporte.
