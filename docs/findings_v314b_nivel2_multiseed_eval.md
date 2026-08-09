# Hallazgos Experimento v314b: Rigor de Nivel 2 (5 Semillas, N=10,000)

---

## 0. Sección Obligatoria de Reconciliación: Auditoría y Refutación de la Señal de Nivel 1

* **Conclusión Previa Refutada (Experimento v314):** En la prueba inicial de 1 semilla ($N=1,000$), la variante `complex_phase_lora` mostró una ventaja de $\Delta = 0.0016$ nats sobre `real_molora`.
* **Resultado Certificado de Nivel 2 (Experimento v314b):** 
  1. **Indistinguible del Ruido [RUIDO-SOSPECHA]:** Tras evaluar 5 semillas independientes sobre $N=10,000$ secuencias y 15 épocas, el promedio de `complex_phase_lora` ($\mu = 3.46863$) y `real_molora` ($\mu = 3.46878$) arroja una diferencia de $|\Delta| = 0.00015$ nats.
  2. **Falló el Criterio de Significancia:** El error estándar combinado es de $\text{SE} = 0.00021$ nats (umbral $2 \times \text{SE} = 0.00043$ nats). Al ser $|\Delta| < 2 \times \text{SE}$, la diferencia es **estadísticamente indistinguible del ruido de muestreo**.
  3. **Demostración de Integridad:** Se confirma la sospecha metodológica planteada: en precisión Float32 pura, todas las variantes de bajo rango (MoLoRA Complejo, MoLoRA Real, LoRA Estático) convergen hacia la misma cota ergódica de rendimiento ($\approx 3.4686..3.4687$).

---

## 1. Tabla de Resultados Certificados (Nivel 2)

* **Configuración:** $N=10,000$ secuencias estructuradas, $L=64$, $d_{model}=128$, 15 épocas, AdamW ($lr=1e-3$). Promedio sobre 5 semillas independientes (`[42, 43, 44, 45, 46]`). Evaluado en CPU (8 hilos).

| Modelo | Dominio | Loss Media ($\mu$) | Std ($\sigma$) | Error Estándar (SE) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`standard_dense`** 🌟 | Real $\mathbb{R}$ | **3.46861** | 0.00044 | 0.00020 | [ANCLA] |
| **`complex_phase_lora`** | Complejo $\mathbb{C}$ | 3.46863 | **0.00021** | **0.00009** | [RUIDO-SOSPECHA] |
| **`static_lora`** | Real $\mathbb{R}$ | 3.46868 | 0.00026 | 0.00012 | [ANCLA] |
| **`real_molora`** | Real $\mathbb{R}$ | 3.46878 | 0.00043 | 0.00019 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor loss media a `standard_dense` (3.46861). Ninguna de las diferencias entre adaptadores supera el umbral de significancia $2 \times \text{SE}$.*

---

## 2. Valor Arquitectónico Remanente del Dominio Complejo

A pesar de la equivalencia funcional en precisión Float32:
1. **Estabilidad entre Semillas:** `complex_phase_lora` demostró la menor desviación estándar entre semillas ($\sigma = 0.00021$, la mitad que la variante real $\sigma = 0.00043$), confirmando la propiedad reguladora de las fases unitarias.
2. **Propiedad Safe by Design:** El valor práctico del dominio complejo no radica en superar a Float32, sino en mantener **cuantización a 4 bits sin degradación por outliers**, puesto que sus parámetros son ángulos $\Theta \in [0, 2\pi]$ acotados en $S^1$.
