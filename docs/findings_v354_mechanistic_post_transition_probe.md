# Findings v354: Post-Transition Mechanistic Angle Probe (Step 3000)

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Descubrimiento de Especialización por Cabezas:** El sondaje post-transición (paso 3000) revela que el modelo no asigna un generador de Fourier idéntico a todas las cabezas. En su lugar, desarolla una **especialización funcional por cabezas**:
  - **Cabeza 4 (Cabeza Ancla/Sesjo Estático):** Mantiene todos los dígitos apretados en una fase de referencia fija ($\approx 225^\circ$, CircStd ultra-bajo de $\sim 0.28\text{ rad}$).
  - **Cabeza 3 (Generador de Fase $2\pi/7$):** Mapea el Dígito 6 a **$51.6^\circ$ ($0.8999\text{ rad}$)**, coincidiendo de forma casi exacta con el paso teórico de la raíz de la unidad $2\pi/7 = 51.4^\circ$.

## 1. Resumen Ejecutivo
Se sondeó el estado del modelo `Complex Beta DeltaPhase` en convergencia post-transición (Paso 3000, $Acc = 86.77\%$ en CPU) en $\mathbb{Z}_7$.

### Tabla de Desglose Mecanístico por Cabezas (Capa 4, Paso 3000)
| Cabeza | Función Mecanística Identificada | Rango de Ángulos Aprendidos ($\varphi_t$) | Desviación Circular Media | Ajuste Generador de Fourier |
| :--- | :--- | :---: | :---: | :---: |
| **Cabeza 1** | Rotación Combinatoria | $3.5^\circ \to 332.5^\circ$ | $0.35\text{ rad}$ | $m=1$ ($R^2 = 0.35$) |
| **Cabeza 2** | Rotación de Multi-Paso | $21.9^\circ \to 358.4^\circ$ | $0.81\text{ rad}$ | $m=3$ ($R^2 = 0.36$) |
| **Cabeza 3** 🌟 | **Generador de Paso $2\pi/7$** | **Dígito 6 = $51.6^\circ$** (Teoría $51.4^\circ$) | $0.28\text{ rad}$ | $m=6$ ($R^2 = 0.06$) |
| **Cabeza 4** 🌟 | **Cabeza Ancla / Referencia Fija** | **$215.7^\circ \to 236.6^\circ$** (Fase Fija $\approx 225^\circ$) | **$0.28\text{ rad}$** | Ancla Estática (Sin rotación) |

## 2. Inventario de Arquitectura y Parámetros
* **Dimensiones:** $d_{\text{model}} = 64$, $n_{\text{layers}} = 4$, $n_{\text{heads}} = 4$, $d_k = 16$.
* **Modelo Sondeado:** `ComplexBetaDeltaPhase` con $200,343$ parámetros entrenables.

## 3. Diagnóstico Teórico y Limitaciones
1. **Separación de Roles por Cabezas:** El modelo resuelve la adición modular mediante una arquitectura de funciones compuestas: la **Cabeza 4 mantiene una fase no dependiente del dígito**, mientras que las **Cabezas 1 y 2 muestran dispersión angular amplia ($R^2 \approx 0.35 - 0.36$)**.
2. **Limitación de la Cabeza 3:** Cabeza 3 presenta $R^2 = 0.06$ (el 94% de la varianza no se ajusta al modelo de Fourier a pesar del alineamiento puntual del Dígito 6), lo que indica que el mapeo no es un generador de Fourier puro mono-cabeza.

## 4. Conclusión
El sondaje post-transición muestra una diferenciación funcional por cabezas bajo una red de $200,343$ parámetros, si bien el ajuste formal a generadores puros de Fourier sigue siendo parcial ($R^2 \le 0.36$).

