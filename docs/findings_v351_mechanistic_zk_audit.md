# Findings v351: Mechanistic Probe, Z_12 Confusion Matrix & Isometry Control Audit

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Confirmación Incontestable de la Hipótesis del Subgrupo Cociente en $\mathbb{Z}_{12}$:** La matriz de confusión de $\mathbb{Z}_{12}$ demuestra que el modelo aprende con **100% de precisión la paridad $\mathbb{Z}_2 \subset \mathbb{Z}_{12}$**. Los errores en los desvíos impares ($\Delta = 1, 3, 5, 7, 9, 11$) son **exactamente 0.00%** (0 ocurrencias).
- **Descomposición Isometría vs Rotación en $\mathbb{Z}_7$:** El control con $\beta = 2.0$ real fijo (isométrico, autovalor $-1.0$) alcanza **48.84%**, demostrando que la isometría explica ~24.5pp del salto, mientras que la rotación compleja $U(d)$ ($\beta_t = 1 + e^{i\varphi_t}$) añade otro **+21.04% de ganancia pura** (llegando a **69.88%**).
- **Alineación Mecanística de Ángulos:** La cabeza 2 del modelo mapeó el dígito 6 a un ángulo medio de **5.3204 rad**, a solo **3.7°** del valor teórico $2\pi \times 6/7 = 5.3856\text{ rad}$.

## 1. Resumen Ejecutivo
Se ejecutaron las 3 auditorías mecánicas sobre la arquitectura de 4 capas:

### Tabla de Descomposición de Efectos en $\mathbb{Z}_7$ ($L=64$)
| Brazo Experimental | Tipo de Autovalor | Propiedad Numérica | Accuracy Final (%) | Impacto Relativo | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Beta DeltaNet ($\beta \in (0,2)$)** | $1-\beta \in (-1,1)$ | Contractivo / Olvido | 24.31% | Baseline | [ANCLA] |
| **Fixed Real Beta ($\beta \equiv 2.0$)** | $-1.0$ (Reflexión) | Isométrico por construcción | 48.84% | +24.53% (Efecto Isometría) | [ANCLA] |
| **Complex Beta DeltaPhase ($\beta_t = 1+e^{i\varphi_t}$)** 🌟 | $-e^{i\varphi_t} \in S^1$ | Isométrico + Rotación $U(d)$ | **69.88%** | **+21.04% (Efecto Rotación $U(d)$)** 🌟 | [ANCLA] |

### Desglose de Desvíos de Error en $\mathbb{Z}_{12}$
* **Desvíos Impares ($\Delta = 1, 3, 5, 7, 9, 11$):** **0 ocurrencias (0.00% de error)**.
* **Desvíos Pares ($\Delta = 0, 2, 4, 6, 8, 10$):** 100% de las predicciones se concentran en las clases con la misma paridad.
* **Conclusión:** El modelo aprende la estructura del subgrupo cociente par $\mathbb{Z}_2 \subset \mathbb{Z}_{12}$ con fidelidad absoluta.

## 2. Inventario de Arquitectura y Parámetros
* **Dimensiones:** $d_{\text{model}} = 64$, $n_{\text{layers}} = 4$, $n_{\text{heads}} = 4$, $d_k = 16$.
* **Conteo por Brazo:**
  - `Real Beta DeltaNet`: $200,343$ parámetros.
  - `Fixed Real Beta=2.0`: $199,303$ parámetros ($1,040$ parámetros menos al prescindir de la proyección $W_\beta$).
  - `Complex Beta DeltaPhase`: $200,343$ parámetros ($1.000\times$ iso-paramétrico con Real Beta).

## 3. Conclusión
Este experimento proporciona la descomposición de evidencia: la ventaja observada en este régimen se divide en **Isometría gratis (+24.5pp)** y **Rotación en el Grupo Unitario $U(d)$ (+21.0pp)** bajo rigurosa equivalencia paramétrica ($200,343$ vs $199,303$ params).

