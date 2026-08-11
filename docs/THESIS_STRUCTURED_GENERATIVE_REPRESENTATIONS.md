# Tesis de investigación: representaciones generativas estructuradas

> Investigo cuándo una transformación neuronal puede aprenderse como una representación estructurada y graduable —en lugar de una tabla densa de pesos—, manteniendo calidad útil con menor coste descriptivo, memoria o cómputo.

## 1. La pregunta de fondo

La tesis no es que una transformada espectral sea un “truco de compresión” para matrices densas. Es que una matriz densa libre es una elección de representación: presupone $d^2$ grados de libertad sin declarar qué estructura genera la transformación.

Para dimensiones grandes, esa elección alcanza un muro físico. Una matriz $10^6\times10^6$ contiene $10^{12}$ coeficientes: aproximadamente 2 TB en fp16, 4 TB en fp32 y del orden de $10^{12}$ MACs por vector. No basta con optimizarla mejor; hay que cambiar el objeto computacional.

La pregunta guía es:

> ¿Qué familias de operadores, geometrías y coordenadas permiten expresar la transformación relevante con un presupuesto graduable de estructura, sin empezar por una tabla de $O(d^2)$ pesos?

## 2. Principio: pesos como síntesis, no como tabla

En vez de aprender directamente $W$, se aprende una descripción que sintetiza $W$ o su actualización:

$$W_K = \mathcal{G}(\theta_K;\,\text{base},\text{geometría},\text{topología}),$$

donde $K$ controla resolución/capacidad. Ejemplos de $θ_K$ son coeficientes espectrales, puntos de control Bézier, centros/radios de conos, gates sobre sustratos congelados, ramas polimórficas, fases complejas o expertos especializados.

La meta es una **curva de capacidad**: con poco $K$, capturar estructura gruesa y filtrar ruido; al ampliar $K$, aproximar transformaciones más ricas y, cuando sea necesario, recuperar la capacidad equivalente a una matriz densa.

## 3. Familias exploradas

| Familia | Representación generativa | Pregunta que explora |
| :--- | :--- | :--- |
| Geometría visual | Trazos Bézier, matchsticks, conos, anillos, muestreo log-polar | ¿La visión puede construirse con primitivas continuas e interpretables, en vez de pesos por píxel? |
| Bases y transformadas | DCT, Walsh/FWHT, Haar, Fourier, FFT causal, espectros multibanda | ¿En qué coordenada la señal, el peso o el update se concentra en pocos modos? |
| Sustratos congelados | Proyecciones aleatorias, bibliotecas Perlin/ruido, gating multiplicativo | ¿Puede la plasticidad vivir en modulaciones de bajo rango sobre una estructura fija? |
| Topología interna | DCT de capas ocultas, permutaciones TSP, cuellos triangulares, neuronas organizadas | ¿Puede imponerse o aprenderse una coordenada donde las variables internas dejen de ser permutables/caóticas? |
| Operadores neuronales | SUM/MAX/L2/Lp, PROD, log, sawtooth, módulos, resonancia | ¿La suma lineal es el agregador adecuado para cada ley o interacción? |
| Fase y complejos | amplitud/fase, atención Hermitiana, FFT causal, interferencia | ¿La fase ofrece un canal natural para posición, periodicidad, memoria y composición? |
| Memoria y expertos | hologramas, PAC/arquetipos, MoE honesto, saliencia | ¿Puede la estructura explícita reemplazar almacenamiento o routing denso? |
| Optimización y estado | DGE, PID, DS, Adam espectral, ARSO | ¿También el optimizador puede representarse/comprimirse estructuralmente? |

## 4. Hallazgos que forman el núcleo de la tesis

### 4.1 Un prior correcto transforma compresión en capacidad útil

Los trazos, matchsticks y conos muestran que unos pocos parámetros geométricos pueden producir filtros visuales interpretables y competitivos en MNIST. DCT/DWT/Walsh y las neuronas espectrales muestran el mismo principio con una base de frecuencias: la resolución se paga en coeficientes, no necesariamente en cada píxel o conexión.

La conclusión no es que toda restricción sea buena. Es que una restricción alineada con la estructura de datos puede actuar simultáneamente como representación, regularización, interpretabilidad y control del paisaje de optimización.

### 4.2 La base no es dogma: debe corresponder al fenómeno

La serie de comparaciones DCT/Walsh, los experimentos de bases congeladas y v87c/v87d muestran una regla: DCT favorece estructura suave/continua; Walsh puede favorecer estructura discreta, binaria o hardware eficiente; una base aleatoria puede ser el prior correcto si el fenómeno está construido en ella.

V331 refuerza el requisito metodológico: una base concreta sólo merece atribución si supera controles ortogonales aleatorios iso-arquitectura. V332/v333 aclaran por qué eso no niega la tesis: cuando el teacher es compresible en una base, la representación coincidente gana en muestras/bits; cuando no lo es, hay que subir el presupuesto $K$.

### 4.3 Capacidad es resolución graduable

V60, V63, V87 y v333 describen la misma curva a distintas escalas. Una matriz espectral truncada no pretende tener la capacidad de una matriz plena con el mismo K; pretende ofrecer una ruta continua:

$$W \approx U^\top C_K V,\qquad K\uparrow \Rightarrow \text{distorsión}\downarrow.$$

En v333, $K=d^2$ recupera exactamente la capacidad de la matriz densa en el experimento lineal. Si el target es compresible, se puede detener mucho antes; si es no estructurado, se paga el presupuesto completo. El problema científico pasa a ser descubrir el sistema de coordenadas y la resolución adecuados.

### 4.4 La agregación y la aritmética son elecciones arquitectónicas

Las neuronas polimórficas, las ramas producto/log/módulo/sawtooth y los MoE honestos exploran que una neurona no tiene por qué ser sólo una suma seguida de activación. Los resultados de generalización estructural sugieren que, para ciertas leyes, incluir el operador correcto puede ahorrar enormes presupuestos frente a aproximarlo mediante muchas unidades lineales.

La contrapartida es optimizabilidad: Lp aprendible, ramas periódicas y algunos operadores discontinuos han mostrado colapsos o paisajes difíciles. Por ello, el operador debe evaluarse tanto por expresividad/OOD como por estabilidad de gradiente y coste vectorizado.

### 4.5 La fase y la geometría compleja son recursos representacionales

La línea compleja/FFT causal indica que amplitud y fase pueden codificar posición, periodicidad y relaciones de interferencia de forma compacta. Los resultados iniciales en texto y MQAR son prometedores, pero requieren los mismos controles de causalidad, fuga, harness y baseline que cualquier otra familia.

### 4.6 El ahorro asintótico exige un camino de implementación

Reducir parámetros no basta. La promesa de una familia estructurada sólo se realiza si la síntesis permite kernels rápidos, sparsity real, transformadas $O(d\log d)$, almacenamiento comprimido y/o estados de optimizador compactos. Materializar $U^\top C V$ como GEMMs densos mide capacidad, pero no entrega el ahorro asintótico.

## 5. Criterio de evidencia

Una propuesta debe evaluarse en varios ejes independientes:

1. **Representación:** curva de error frente a resolución $K$, rank, core, número de primitivas o expertos.
2. **Información:** parámetros efectivos, bits de valores, escalas, soporte, permutaciones y coste de metadata; curvas rate–distortion, no sólo ratio nominal.
3. **Cómputo:** coste asintótico y coste medido en FLOPs, memoria pico, tiempo de síntesis, latencia online y tras fusionar/compilar.
4. **Generalización y atribución:** test retenido, OOD cuando aplica, controles de base aleatoria, ablations, semillas, etc. 
5. **Interpretabilidad**: visualización de representaciones, número de nodos activos, relación con conceptos semánticos.
6. **Estabilidad**: estabilidad frente a ruido, cambios pequeños en parámetros, diferentes semillas, tasas de aprendizaje, diferentes precisiones.
7. **Seguridad**: resistencia a ataques adversariales.
8. **Eficiencia**: speedups reales en kernel, compresión de checkpoint y latencia online/offline.
9. **Uso de hardware:** aprovechamiento de operaciones tensoriales, SIMD, memoria caché y paralelismo.
10. **Portabilidad**: comportamiento en arquitecturas y plataformas distintas.


Una victoria en un eje no implica las otras. Por ejemplo, v333 demuestra una curva de capacidad/bits sintética; no prueba speedup de kernel ni mejora en lenguaje. V330/v331 limitan los claims de base espectral fija en Tiny Shakespeare, pero no invalidan la hipótesis más amplia de representación generativa estructurada.

## 6. Programa de escalado

El itinerario coherente es:

1. **Medir compresibilidad:** espectro, topología, TV, sparsity, energía por bandas y estabilidad de soporte.
2. **Elegir/descubrir coordenadas:** bases conocidas, permutaciones, estructuras geométricas, bases aprendibles/butterfly o routing condicionado por datos.
3. **Asignar presupuesto adaptativo:** K, bits, bandas, expertos y resolución donde la distorsión lo justifique.
4. **Compilar la estructura:** kernels FHT/FFT/wavelet, sparse updates, cuantización por bandas y formatos de checkpoint.
5. **Validar en tareas reales:** corpus/tokenizador real, visión natural, contextos largos y escalas que hagan imposible el baseline $O(d^2)$ directo.

## 7. Formulación compacta

> La inteligencia artificial escalable no tiene por qué aprender una tabla de conexiones. Puede aprender una receta: una geometría, una base, una fase, una familia de operadores y un presupuesto de resolución. La hipótesis se sostiene sólo cuando esa receta describe la tarea con menos información, menos cómputo y calidad verificable frente a controles equivalentes.
