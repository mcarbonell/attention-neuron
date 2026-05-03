# Hallazgos V214-V217: La Batalla de los Espacios (Lineal vs Logarítmico)

## Objetivo
Determinar si una red neuronal puede aprender de forma autónoma a elegir la representación matemática (Lineal o Logarítmica) más eficiente para resolver operaciones aritméticas mixtas.

## Evolución de la Arquitectura
1.  **V214 (Soft MoE)**: Colapso total. Los expertos "colaboran" para anular sus errores en lugar de especializarse. La red miente para bajar la pérdida localmente.
2.  **V215-V216 (Hard/Warmup MoE)**: Primeros indicios de especialización forzada mediante Gumbel-Softmax. Descubrimos que el Router es propenso a prejuicios prematuros.
3.  **V217 (Honest MoE)**: Selección natural pura basada en el error mínimo instantáneo. Los expertos compiten por el derecho a recibir gradiente.

## Mapa de Verdad Matemático (V217 Final)
Tras someter a la red a rangos extremos de datos ($x \in [1, 1000]$), los expertos se repartieron el dominio de la siguiente manera:

| Operación | Ganador Indiscutible | Razón del Éxito |
| :--- | :--- | :--- |
| **Multiplicación (*)** | **LOGARÍTMICO** | En Log-Space es una simple suma ($\log x + \log y$). Imbatible. |
| **Resta (-)** | **LINEAL** | El espacio Log (vía `exp`) no puede producir números negativos. |
| **Suma (+)** | **LOGARÍTMICO** | Sorprendente preferencia. Sugiere que la red prefiere la precisión relativa del Log-Space incluso al coste de aprender `Log-Sum-Exp`. |
| **División (/)** | **LINEAL** | **Paradoja**: Aunque es una resta en Log-Space, el MLP prefiere aproximar linealmente $1/y$ para evitar el ruido de las transformaciones `log`/`exp`. |

## Conclusiones Finales de la Sesión
1.  **Arquitectura como Destino**: La forma en que representamos los datos (Input Mapping) es más poderosa que cualquier aumento en el número de capas. Una red de 64 neuronas en Log-Space destruye a una red de 1000 neuronas en espacio Lineal para tareas multiplicativas.
2.  **El Experto Honesto**: La mejor forma de forzar especialización en IA no es entrenar un enrutador inteligente, sino crear una competición darwiniana donde solo el que tiene menos error sobrevive (recibe gradiente).
3.  **Límites de la Precisión**: El hecho de que la división se mantenga en el espacio lineal sugiere que las funciones trascendentales (`log`/`exp`) introducen un "suelo de ruido" que las redes intentan evitar si existe una alternativa lineal aceptable.

---
**Commit Final**: `V217_final_honest_moe_mapping`
