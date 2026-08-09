# Hallazgos Experimento v308: Dynamic Multiplicative Low-Rank Adaptations (Fase 1)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En la planificación de la línea de investigación (`RESEARCH_LINE_DYNAMIC_LOW_RANK.md`), se planteó que la modulación dinámico-multiplicativa de entrada/salida $y = \sigma(g_{out}(x)) \odot (W_0 ( \sigma(g_{in}(x)) \odot x ))$ serviría como baseline ligero inicial para evaluar expresividad.
* **Resultado del Experimento v308:** La loss del modelo dinámico multiplicativo no residual (4.1476) quedó atascada en el nivel de entropía teórica de la tarea ($\ln 64 \approx 4.1588$), mostrando un rendimiento marginalmente inferior al modelo denso baseline (4.1348) y un `internal_overhead_time` significativo (2.35s frente a 0.08s).
* **Diagnóstico de Reconciliación (Checklist de 5 causas):**
  1. **Bug/Artefacto de inicialización:** Multiplicar dos compuertas $\sigma(g_{in}) \times \sigma(g_{out})$ con pesos inicializados cerca de 0 resulta en una atenuación de la señal por un factor de $0.25$ por capa ($0.25^2 = 0.0625$ en 2 capas), sofocando los gradientes iniciales.
  2. **Harness sintético i.i.d.:** La tarea de prueba utilizó una rotación de tokens sintéticos i.i.d. aleatorios uniformes (`torch.roll`), la cual carece de patrón computable determinante.
  3. **Ausencia de conexión residual:** Sin conexión de atajo $x + \text{Layer}(x)$, las compuertas multiplicativas colapsan el flujo de la señal.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=1000$ secuencias, $L=64$, $d_{model}=128$, $r=16$, 5 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Wall Clock (s) | Internal Overhead (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`standard_dense`** 🌟 | **49,216** | **4.1348** | **1.13** | **0.08** | **0.0515** | [SEÑAL] |
| **`dynamic_low_rank`** (v308) | 65,600 | 4.1476 | 4.91 | 2.35 | 0.0501 | [CIERRE-PREMATURO-SOSPECHA] |
| **`static_lora`** | 57,408 | 4.1533 | 2.02 | 0.08 | 0.0506 | [SEÑAL] |

*Nota: El marcador 🌟 se asigna estrictamente al menor valor numérico de Loss (4.1348).*

---

## 2. Análisis del Desempeño y Coste

1. **Rendimiento Causal/Loss:** Los 3 modelos (Tabla 1) se mantienen en la vecindad de la entropía máxima ($\ln 64 \approx 4.1588$). El modelo `standard_dense` obtiene 4.1348 frente a 4.1476 de `dynamic_low_rank`. La diferencia de $\Delta = 0.0128$ nats es atribuible a la atenuación inicial por la doble sigmoide de gating sin residuales.
2. **Internal Overhead:** El modelo `dynamic_low_rank` gastó 2.35 segundos de overhead interno (50% del tiempo total de ejecución), debido a la invocación secuencial de 4 sub-capas lineales (`nn.Sequential(Linear, SiLU, Linear)`) por cada pase dinámico en CPU.

---

## 3. Amenazas a la Validez

1. **Validez Interna (Inicialización):** La función sigmoide inicializada en 0 atenúa la señal en un 75% por capa. En la versión `v308b` debe usarse gating centrado en la identidad ($\text{sigmoid} + 0.5$ o $1.0 + \text{tanh}$).
2. **Validez Externa (Tarea Sintética):** La secuencia sintética con permutación aleatoria i.i.d. no evalúa capacidad de memoria asociativa real (como MQAR).
3. **Validez Arquitectónica:** No se incluyeron conexiones residuales ni LayerNorm, favoreciendo a las capas lineales densas simples sobre las compuertas profundas.
