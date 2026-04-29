# BLUEPRINT: El Hipocampo Holográfico (Memoria de Contexto Infinito)

## 1. El Muro del Contexto en los LLMs
Actualmente, los modelos Transformers escalan la memoria de forma estrictamente espacial (mediante la Caché KV). Guardar más "pasado" requiere hacer la ventana más larga, lo que aumenta linealmente (y cuadráticamente en cálculo) el consumo de RAM. Al llegar a contextos masivos (100k-1M de tokens), el modelo choca contra el "Memory Wall" y el hardware colapsa por falta de memoria VRAM.

## 2. La Solución de Resonancia: Memoria Espectral O(1)
Inspirados en la neurobiología y el procesamiento de señales, proponemos almacenar la **"onda narrativa"** usando un Tensor Global de Memoria (El Hipocampo) de tamaño constante, independientemente de si la conversación dura 10 minutos o 10 años.

### Mecánica (Chunking & FWHT)
1. **Flujo de Trabajo Temporal:** La red lee tokens en pequeños bloques fijos (ej. $C=512$ tokens), que actúan como la *Memoria a Corto Plazo* del córtex.
2. **Transformada de Consolidación (El "Sueño"):** Cuando el bloque está lleno, aplicamos la Transformada Rápida de Walsh-Hadamard (FWHT) a lo largo del eje del *tiempo*.
   - Las **Altas Frecuencias** representan palabras exactas, signos de puntuación y ruido temporal.
   - Las **Bajas Frecuencias** representan el sentido global o "arquetipo" narrativo del fragmento.
3. **Consolidación Holográfica:** Descartamos las altas frecuencias (olvido selectivo). Multiplicamos las bajas frecuencias por una "firma de fase" ortogonal (para saber a qué momento macro pertenece) y las **sumamos (interferencia constructiva)** al Tensor Global del Hipocampo.
4. **Recuperación (Resonancia):** Para buscar un recuerdo perdido en el pasado, la red envía un vector "Query" al Hipocampo. El sistema multiplica la consulta espacial y encuentra la resonancia temporal con las frecuencias guardadas. Extrae la "firma" de ese momento y la usa para aislar el contexto original (el Value) filtrando el ruido del resto de la conversación.

## 3. Experimento V88: La Aguja en el Pajar Continuo
Para validar empíricamente este concepto extremo sin usar VRAM masiva, diseñamos una prueba sintética rigurosa:
- **Secuencia (Streaming):** Flujo continuo de decenas de miles de pasos (ej. 51,200 tokens de ruido de fondo simulado).
- **El Evento ("La Aguja"):** En un momento aleatorio, inyectamos una "clave" y un "valor" representados por vectores. Emulando a la **Amígdala**, este evento recibe un pico de amplitud (salience) para destacar su importancia sobre la charla trivial.
- **La Búsqueda:** Al final del streaming, inyectamos la "clave" (Query) en el sistema.
- **El Reto:** El Hipocampo debe ser capaz de reconstruir el "valor" exacto usando **solo un pequeño tensor espectral constante**, habiendo procesado toda la secuencia sin guardar en ningún momento el historial de los vectores reales en RAM.

## 4. Impacto Esperado
Si este experimento tiene éxito y logramos aislar la señal del ruido, habremos creado la primera prueba de concepto de un **RAG Interno Endógeno**. Un sistema donde la IA recuerda por "interferencia de ondas semánticas" en un espacio $O(1)$, destruyendo para siempre el límite de la ventana de contexto.