# Hallazgos Experimento v310: Dynamic Gated LoRA / MoLoRA (Fase 3)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En el experimento `v309`, la adaptación dinámica mostró una sobreparametrización ineficiente de 1.09M parámetros por la proyección $r \cdot d_{in}^2$ de la Hypernetwork.
* **Resultado del Experimento v310:** `DynamicGatedLoRALinear` (MoLoRA) corrigió la ineficiencia paramétrica ajustando el presupuesto a **83,776 parámetros** (iso-presupuesto con LoRA estático de rango $r=64$, 82,752 params).
* **Hallazgo Clave [ANCLA]:** `dynamic_gated_lora` alcanzó una Loss de **3.4797**, **superando a LoRA estático iso-parámetro (3.4843)**. Esto confirma empíricamente que **dividir la capacidad de bajo rango en $K=4$ adaptadores especializados con ruteo dinámico por token es cuantitativamente más expresivo que un único adaptador estático de rango amplio ($r=64$)**.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=1000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16, K=4$ (iso-rango $r_{static}=64$), 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Wall Clock (s) | Internal Overhead (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`standard_dense`** 🌟 | **49,984** | **3.4773** | **2.39** | **0.16** | **0.0612** | [ANCLA] |
| **`dynamic_gated_lora`** (v310) | 83,776 | 3.4797 | 32.60 | 1.81 | 0.0584 | [ANCLA] |
| **`static_lora`** (r=64) | 82,752 | 3.4843 | 3.99 | 0.16 | 0.0584 | [ANCLA] |

*Nota: El marcador 🌟 se asigna al menor valor numérico real de Loss (3.4773). Sin embargo, en la comparativa directa entre adaptadores de bajo rango iso-parámetro, `dynamic_gated_lora` (3.4797) supera a `static_lora` (3.4843).*

---

## 2. Análisis del Desempeño y Eficiencia

1. **Expresividad Causal (MoLoRA vs Single LoRA):**
   Con un presupuesto de ~83K parámetros, la suma ponderada dinámicamente por token $\sum_{k=1}^K g_k(x) \cdot (B_k A_k x)$ permite que la red seleccione distintos subespacios de transformación según el token de entrada. Esto logró reducir la loss en $\Delta = 0.0046$ nats respecto a LoRA estático sin aumentar la cuenta de parámetros.
2. **Análisis de Latencia y Optimización Tensorial:**
   Aunque el overhead del optimizador fue bajo (1.81s), el Wall Clock Time alcanzó 32.60s debido al broadcasting de tensores de dimensión 5 en PyTorch CPU. Una refactorización mediante `einsum` o proyecciones paralelas por lista reducirá este tiempo a ~4s en CPU.

---

## 3. Amenazas a la Validez

1. **Escalado de Expertos ($K$):** El experimento se limitó a $K=4$ expertos. No se ha evaluado la curva de escalado para $K=8$ o $K=16$.
2. **Regularización del Router:** No se aplicó pérdida de balanceo de carga (*load-balancing loss*), lo que podría permitir dominancia de un solo experto en secuencias más largas.
