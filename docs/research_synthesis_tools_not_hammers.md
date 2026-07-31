# Síntesis de investigación: herramientas adecuadas para cada problema

## Idea central

La trayectoria del repositorio no apunta a encontrar una única arquitectura que sustituya a todas las demás. Apunta a una tesis más interesante:

> Una red eficiente debería reconocer qué estructura tiene el problema y seleccionar la representación, el operador, la memoria y el presupuesto de cómputo adecuados.

> Las matrices densas son un martillo universal: flexibles, pero caras y poco informativas. Los experimentos del repositorio exploran herramientas especializadas para distintos regímenes.

---

## 1. Mapa de herramientas

| Régimen | Herramientas | Función |
|---|---|---|
| Estructura espacial local | CNN, Cone neurons, Haar | Localidad, bordes, invariancia |
| Estructura espectral | DCT, Walsh, Fourier | Compresión y mezcla estructurada |
| Periodicidad y oscilación | Resonancia, fase | Sintonía e interferencia |
| Interacciones multiplicativas | Espacio logarítmico, PROD | Convertir productos en operaciones simples |
| Discontinuidades | DGE, ramas simbólicas | Evitar gradientes engañosos |
| Recall asociativo | DeltaPhase holográfico | Memoria dependiente del contenido |
| Novedad y familiaridad | Atlas espectral, memoria de contraste | Saber cuándo el caso es conocido |
| Cómputo adaptativo | Surprise gate, MoE de competencia | Gastar recursos solo cuando hace falta |
| Hardware extremo | Ternarización, inhibición, GAP | Reducir multiplicaciones y memoria |
| Dinámica de entrenamiento | PID, stage gating | Separar exploración y refinamiento |

---

## 2. Los hallazgos más profundos

### DeltaPhase: memoria asociativa con fase y regla delta

V298–V299 son actualmente la línea experimental más fuerte. La contribución potencial no es simplemente usar números complejos, sino combinar claves fasoriales, memoria matricial, actualización delta basada en error residual y estado constante respecto a la longitud de secuencia.

En V298, DeltaPhase alcanza 99.95% en MQAR, igualando a la atención causal del benchmark. En V299, bajo presupuesto de estado igualado, mantiene 95.98% a 64 pares frente a 73.14% para DeltaNet real.

La hipótesis mecanística más interesante es que la fase compleja hace más uniforme la geometría de las claves y reduce la diafonía causada por normas variables.

El ablation crucial es comparar DeltaPhase complejo, Delta real normalizado, Delta real sin normalizar, Delta complejo con magnitud aprendida y regla delta frente a actualización Hebbiana.

### Fase como principio común

La fase aparece en resonancia, filtros temporales, binding asociativo e invariancia geométrica. La hipótesis unificadora podría ser:

> La fase codifica relaciones relativas: posición, diferencia temporal, compatibilidad entre claves o sintonía con una frecuencia.

La pregunta precisa no es si la fase es siempre superior, sino qué tipo de relación expresa mejor que una amplitud real.

### La arquitectura debe respetar la geometría del dominio

V55, V97, V101–V104, V117 y V258–V259 apuntan a la misma regla: la estructura del sustrato debe ser compatible con la geometría de los datos. En imágenes importa la localidad; en secuencias, el orden; en funciones periódicas, la fase; en memoria, el binding; en OOD, la estructura de la ley.

Una base eficiente pero incompatible con el dominio puede ser peor que una base más simple.

---

## 3. Representaciones estructuradas y compresión

V63 demuestra que un MLP completo puede comprimirse radicalmente mediante núcleos DCT, conservando 97.59% en MNIST con 11.9k parámetros frente a 670k. V66 extiende la idea a todas las proyecciones de un pequeño Transformer.

V229–V236 muestran que la poda y cuantización espectral pueden preservar información crítica mejor que métodos puramente espaciales. Pero V235–V236 corrigen una intuición demasiado simple: la información no está solo en las bajas frecuencias. La poda por magnitud no es monótona y existe un umbral de estabilidad dependiente del dominio.

V239 añade otra lección: exponer todos los componentes espectrales individuales puede ser peor que entregar la suma interferente. La compresión puede actuar como síntesis de señal, no solo como ahorro de memoria.

V171–V189 y V202–V203 muestran que las bases armónicas pueden resolver tareas con muy pocos parámetros, especialmente cuando la inicialización está ordenada. El punto importante de V186 es que la inicialización determina la geometría del espacio de búsqueda.

V101–V104 muestran dos comportamientos diferentes: ConeAttn aprende radios crecientes con la profundidad, mientras ConeFFN colapsa a una o dos dimensiones y se comporta como selección sparse de features.

---

## 4. Leyes, composición y extrapolación

V133–V134 combinan primitivas explícitas —producto, división y bases espectrales— para aproximar funciones con cientos de parámetros. V190 y V192–V193 muestran que la estabilidad OOD puede ser más importante que el error local.

V246–V249 llevan esto hacia el descubrimiento de leyes:

1. expandir el input con una biblioteca de bases;
2. ajustar una combinación pequeña;
3. aplicar poda agresiva;
4. leer la fórmula resultante;
5. componer leyes en varias capas.

La poda es esencial. Sin ella, las bases redundantes o inestables contaminan la extrapolación. Con ella, la red puede convertirse en un descubridor de fórmulas locales.

V213 muestra que un gradiente sintético correcto no basta para entrenar funciones discontinuas. V240 encuentra una solución modular: Adam para lo continuo, DGE para unos pocos parámetros simbólicos y un MoE para combinarlos.

> No todo el sistema debe ser diferenciable; cada subproblema debe recibir el algoritmo de búsqueda que corresponde a su geometría.

---

## 5. Selección de herramientas y control del cómputo

V214 muestra colusión entre expertos. V217 usa competición directa por error y obtiene especialización por espacios matemáticos: logarítmico para ciertas operaciones, lineal para otras. V218 compone `Log → Harmonic` para `sin(x·y)` y generaliza mejor OOD que un camino lineal-MLP.

Esto sugiere que el router debe seleccionar cadenas de representación y operación, no solo expertos aislados.

V219 fracasa al entrenar una cabeza de confianza: fuera de distribución se vuelve más confiada mientras se equivoca más. V220 encuentra una señal mejor mediante un atlas espectral de familiaridad. V221 usa esa señal para abstenerse y obtiene 100% de precisión sobre los ejemplos aceptados, aunque con una tasa de rechazo alta.

La confianza robusta debería apoyarse en memoria estructural, distancia o consistencia, no únicamente en una predicción de confianza producida por la misma red.

V135 muestra una rama analítica rápida y una rama espectral lenta activada por sorpresa. Para operaciones fáciles, el cómputo lento casi se apaga; para funciones difíciles, se activa. Esto es una forma concreta de test-time compute adaptativo.

---

## 6. Gating, sustratos congelados e inhibición

V253–V257 muestran que pesos ternarios congelados más gates aprendibles pueden alcanzar resultados útiles con muy pocos parámetros entrenables. V254 establece que la inhibición negativa es esencial: los pesos `{0,1}` acumulan sesgo positivo y no representan bien contraste ni bordes.

V255 elimina también las multiplicaciones mediante gates ternarios, pagando una caída importante de accuracy. Esto define un compromiso entre eficiencia de entrenamiento, memoria, inferencia y calidad final.

V291 sugiere que muchas tareas se resuelven mediante una subred efectiva menor que el backbone disponible. La interpretación de un atractor universal de sparsidad requiere todavía controles por distribución, arquitectura y presupuesto real.

---

## 7. Memoria, novedad y estado

V88 introduce memoria holográfica de tamaño constante y recupera una señal dentro de un stream largo, aunque necesita amplificación artificial de la señal crítica.

V250 muestra que la memoria de contraste puede ayudar a detectar novedad, pero destruir información cuando los ejemplos llegan agrupados por clase.

La combinación más prometedora separa memoria positiva, memoria de contraste y memoria de familiaridad. DeltaPhase parece adecuado para la primera. El Atlas espectral puede cubrir la tercera. La segunda necesita un mecanismo adaptativo, no una resta fija.

---

## 8. Optimización como herramienta especializada

V261–V273 muestran que PID puede superar a Adam en ciertos regímenes y que cambiar de fase —alta inercia durante exploración, alta amortiguación durante refinamiento— produce saltos significativos.

Esto conecta con stage gating: descubrir estructura mediante gates, congelarla, refinar pesos o expertos por turnos y aplicar una fase final conjunta opcional.

La idea transversal es separar descubrimiento y refinamiento. Entrenarlo todo simultáneamente puede destruir la estructura que el sistema acaba de descubrir.

---

## 9. Arquitectura modular que emerge

```text
entrada
  ↓
análisis de geometría, familiaridad y dificultad
  ↓
selector de representación
  ├── espacial / convolucional
  ├── espectral / DCT / Walsh
  ├── fase / resonancia
  ├── logarítmica / analítica
  └── simbólica / discreta
  ↓
selector de operador
  ├── mezcla local o convolución
  ├── operador resonante
  ├── DeltaPhase para recall
  ├── experto polimórfico
  └── experto simbólico
  ↓
control de esfuerzo
  ├── ruta rápida
  ├── más memoria
  ├── reflexión espectral
  ├── segundo experto
  └── abstención
  ↓
salida + familiaridad + estimación de error
```

Las herramientas pueden ocupar distintos niveles: representación, operador, memoria, gate y verificador.

---

## 10. Preguntas prioritarias

### A. ¿Qué aporta realmente la fase compleja?

Comparar DeltaPhase complejo contra Delta real normalizado, a igual estado, con claves correlacionadas y cargas crecientes.

### B. ¿Puede el router elegir cadenas completas?

Comparar `Lineal → MLP`, `Log → Harmonic`, `DCT → Spectral` y `Phase → DeltaMemory` midiendo error, coste y extrapolación.

### C. ¿La confianza debe ser geométrica?

Combinar Atlas de familiaridad, error de reconstrucción y consistencia entre expertos. Evaluar abstención a cobertura fija.

### D. ¿La memoria debe ser positiva, contrastiva o ambas?

Comparar memoria acumulativa, DeltaPhase, memoria de novedad y memoria dual con decay aprendido.

### E. ¿La especialización sobrevive a tareas reales?

Probar perturbaciones, dominios cambiantes, tareas composicionales y lenguaje natural; no solo regiones artificiales.

### F. ¿Qué es barato realmente?

Medir por separado parámetros entrenables, parámetros congelados, memoria de estado, FLOPs, tiempo real y energía.

---

## 11. Principios adicionales descubiertos en las eras tempranas

### Sustratos ricos y ruido estructurado

V16–V33 establecen la intuición original del proyecto: una gran biblioteca de pesos congelados puede funcionar como un diccionario de rasgos, mientras una modulación pequeña aprende cómo acceder a él.

V24 muestra que mezclar varios sustratos aleatorios puede ser más eficaz que aumentar el rango de uno solo. V26 añade que el ruido estructurado —Perlin a distintas escalas— supera ligeramente al ruido blanco y produce una jerarquía interpretable: frecuencias altas en las primeras capas y bajas en las profundas.

La conclusión no es que el ruido sea mágicamente bueno, sino que un sustrato fijo puede aportar un prior útil si contiene la geometría adecuada. El dial aprendible selecciona y combina; no necesita inventar cada detector desde cero.

### La vía continua tiene un papel específico

V29–V30 prueban splats gaussianos y ventanas suaves como extractores espaciales continuos. Su rendimiento queda por debajo de CNN/Walsh, pero el patrón de fallo es informativo: capturan bien forma global y posición, pero pierden textura y alta frecuencia.

Esto sugiere una división natural:

- visión continua para estructura macro y resolución variable;
- filtros locales o espectrales para detalles finos.

No hay que pedir a un extractor continuo que resuelva el trabajo de una convolución local. Puede ser la primera herramienta de una cadena híbrida.

### Frecuencia y topología son complementarias

V110 combina firmas de islas con Walsh y obtiene más de 93% con 1.3k parámetros. V118 consigue invariancia rotacional casi perfecta descartando fase, pero pierde orientación y baja a aproximadamente 62%. V119 recupera accuracy añadiendo caminos estructural y orientacional, aunque aparece conflicto a rotaciones grandes.

Este arco revela un principio muy general:

> La invariancia perfecta puede destruir la información que la clasificación necesita.

La solución no es elegir entre invariancia y detalle, sino mantener rutas separadas y dejar que un gate decida cuánto confiar en cada una.

### PAC: la transformada no siempre aporta la ventaja

La auditoría de V138–V143 contiene una corrección importante. Si solo se aplica una transformación ortogonal y después se usa distancia L2 o producto escalar completo, el ranking es equivalente al del dominio original por Parseval.

Por tanto, en esos experimentos la contribución real está más cerca de PAC, los arquetipos y la memoria que de Walsh en sí. La base espectral empieza a aportar algo específico cuando se combina con truncación, filtrado, cuantización o una operación no invariante a cambios de base.

Este es un principio metodológico valioso para todo el proyecto:

> Una reparametrización ortogonal no es por sí sola una nueva capacidad; la capacidad aparece cuando la arquitectura trata de forma distinta las coordenadas transformadas.

### Neurogénesis por error

V167–V170 representan quizá la conexión más directa entre los experimentos y una arquitectura auto-organizada. La red añade capas residuales de especialistas solo para corregir los errores que sobreviven a las capas anteriores.

La idea es distinta de simplemente hacer una red más profunda:

1. entrenar una solución inicial;
2. localizar errores recurrentes;
3. añadir una herramienta especializada para ese residuo;
4. congelar o estabilizar lo anterior;
5. repetir si todavía queda una clase de error estructurada.

V170 alcanza 96.08% con cuatro capas y mejora progresiva, mientras V168 muestra que una modulación dinámica sin la organización jerárquica adecuada puede empeorar.

Esta es probablemente la forma más natural de conectar el catálogo completo con la idea del “taller”: el sistema no solo elige entre martillos ya disponibles; puede fabricar un utensilio nuevo para el tipo de clavo que todavía no sabe resolver.

### Escala teórica frente a coste real

V168 también deja una advertencia práctica. La factorización y el gating jerárquico pueden ofrecer una capacidad teórica enorme, pero el coste real depende de vectorización, memoria intermedia y lanzamientos de kernels.

La versión vectorizada mejora 7.5x y el MoE jerárquico evita matrices de activación gigantes. Esto conecta con la necesidad de medir siempre cuatro cosas por separado: capacidad representacional, parámetros entrenables, memoria de estado y tiempo/energía reales.

---

## Conclusión

La contribución potencial del proyecto no es una sustitución universal de las redes densas. Es una visión de arquitectura heterogénea:

> Una inteligencia eficiente debería reconocer qué clase de problema tiene delante, escoger una representación compatible, aplicar el operador adecuado, recordar solo lo necesario y gastar más cómputo únicamente cuando la situación lo exige.

DeltaPhase parece actualmente la mejor candidata para el componente de memoria asociativa. Pero el sistema completo probablemente necesite también bases espectrales, expertos analíticos, resonancia, memoria de familiaridad, abstención y control adaptativo del esfuerzo.

El siguiente salto no sería encontrar un martillo más potente, sino construir un buen taller.
