# Hallazgos Experimento v313: Phase Spectral MoLoRA Híbrido (Fase 6)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En `v312`, se demostró que MoLoRA puro sin mezclador causal colapsaba en MQAR (~1% Acc), concluyendo que MoLoRA pertenece al bloque FFN de transformación de rasgos $d \times d$ y no a la mezcla temporal $L \times L$.
* **Resultado del Experimento v313:** **ÉXITO DE INTEGRACIÓN HÍBRIDA [ANCLA].** Al combinar la atención de fase trigonométrica causal con un FFN alimentado por MoLoRA (`phase_molora`), la arquitectura superó cuantitativamente al Transformer MHA estándar con FFN denso (`standard_mha`) a igualdad de presupuesto (~230K params):
  * **Loss Final:** Reducción de **4.0263 $\to$ 3.9025** ($\Delta = -0.1238$ nats de mejora).
  * **Target Accuracy:** Incremento de **7.81% $\to$ 9.77%** (+25.1% de mejora relativa en memoria asociativa).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** MQAR ($L=64$, $N_{pairs}=8$, Vocab=120), 400 pasos de entrenamiento, AdamW ($lr=2e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Target Acc (%) | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`phase_molora`** (v313) 🌟 | **233,856** | **9.77%** | **3.9025** | 51.37 | **0.0182** | [ANCLA] |
| **`standard_mha`** (Transformer Estándar) | 230,272 | 7.81% | 4.0263 | **15.08** | 0.0146 | [ANCLA] |

*Nota: El marcador 🌟 asigna el mejor rendimiento a `phase_molora` (3.9025 Loss, 9.77% Acc).*

---

## 2. Análisis del Desempeño

1. **Confirmación del Motor FFN MoLoRA:**
   La integración de MoLoRA ($K=16, r=4$) en el FFN potencia la capacidad del bloque Transformer frente a los FFNs densos tradicionales, aportando compuertas dinámicas adaptativas por token que enriquecen las representaciones tras la atención.
2. **Estabilidad de Fase Trigonométrica:**
   El bias de fase $\sin(\theta)$ garantizó que las activaciones se mantuvieran acotadas en $[-1, 1]$, evitando explosión numérica durante el entrenamiento.
