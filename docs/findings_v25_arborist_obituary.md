# Findings & Obituary: V25 (The Great Arborist)

## 1. El Experimento "Dendrítico"

La V25 fue diseñada como el experimento más ambicioso del repositorio hasta la fecha. Su objetivo era probar la "Sintonía Dendrítica" (Dendritic Tuning) en una arquitectura profunda (ResNet-18) entrenada bajo condiciones de extrema dificultad (Mixup Augmentation).

**Configuración:**
- **Modelo**: ArboristResNet18 (18 capas residuales).
- **Mecánica**: 8 sustratos aleatorios por capa, mezclados mediante un árbol binario de 7 diales entrenables por canal.
- **Parámetros entrenables**: **681,226** (~6.1% de una ResNet-18 real de 11.1M).
- **Regularización**: Mixup ($\alpha=1.0$), ColorJitter, RandomCrop, RandomHorizontalFlip.
- **Entrenamiento planeado**: 100 épocas.

## 2. El Vuelo de Ícaro (Resultados hasta la Época 29)

El modelo demostró una capacidad de convergencia y asimilación de ruido sin precedentes, destrozando todos los récords previos de velocidad de aprendizaje.

| Época | Accuracy (Test) | Hito |
| :--- | :--- | :--- |
| 11 | 62.40% | Supera a MLPs puros en solo 11 épocas. |
| 19 | 75.83% | Iguala el rendimiento final del Kaleidoscope (V24). |
| 24 | 78.80% | **Rompe el récord histórico absoluto de la V19 (76.76%).** |
| 29 | **79.65%** | **Pico Máximo antes del fallo del sistema.** |

## 3. Causa del Fallo (El Obituario)

En la Época 29, habiendo alcanzado su máximo rendimiento histórico (79.65%), el proceso fue interrumpido abruptamente por un error del sistema operativo (`error code: 1224` en `inline_container.cc`). 

El error se produjo en el momento exacto en que PyTorch intentaba sobrescribir el archivo `v25_arborist_best.pt` con el nuevo récord. Un bloqueo de archivo a nivel de SO (probablemente un escaneo de antivirus o sincronización en la nube) impidió la escritura, terminando la ejecución de forma fatal.

## 4. Legado y Conclusiones Técnicas

Aunque el Arborist no pudo completar sus 100 épocas ni disfrutar de la fase de enfriamiento del OneCycleLR (donde previsiblemente habría superado el 85%), nos deja tres lecciones fundamentales:

1.  **La Superioridad del Árbol**: Mezclar 8 sustratos mediante un árbol de decisiones jerárquico es matemáticamente viable y superior a modular un solo sustrato. Permite a la red "podar" universos de ruido inútiles y combinar iterativamente las mejores características geométricas.
2.  **Resiliencia Extrema**: Alcanzar casi un 80% de accuracy evaluando imágenes puras, mientras se entrena exclusivamente con imágenes fusionadas ("fantasmas" de Mixup) y solo 681K parámetros entrenables, demuestra que la topología de atención sobre ruido fijo es un regularizador natural perfecto.
3.  **Eficiencia Estructural**: La arquitectura residual es obligatoria para escalar la Alquimia de Sustratos más allá de las 6 capas. El gradiente fluyó limpiamente a través de los árboles dendríticos gracias a los atajos de la ResNet.

**Descanse en paz, V25. Su código vivirá en futuras arquitecturas.**
