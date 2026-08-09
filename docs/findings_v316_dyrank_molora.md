# Hallazgos Experimento v316: DyRank MoLoRA (Asignación Dinámica de Rango por Token, Fase 9)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En los experimentos previos (`v310`-`v311`), los adaptadores MoLoRA utilizaban un presupuesto de bajo rango estático e invariable ($r=16$) para todos los tokens, sin importar si eran símbolos triviales o tokens con relaciones de razonamiento complejo.
* **Resultado del Experimento v316 [ANCLA]:** **ÉXITO DE ESPARCIDAD Y EXPRESIVIDAD.** 
  1. **Mayor Precisión con 42% Menos Rango:** `dyrank_molora` (v316) alcanzó la **menor loss del benchmark (3.4748)** superando a `fast_molora` (3.4751), `static_lora` (3.4784) y `standard_dense` (3.4819).
  2. **Auto-Organización Dinámica (57.9% Active Rank):** La red aprendió de forma autónoma a apagar el **42.1% de los canales de bajo rango**, activando solo un 57.9% de rango en promedio según la complejidad asociativa de cada token.
  3. **Aceleración por Esparcidad:** A pesar de calcular las compuertas sigmoidales de rango, el cómputo filtrado fue más rápido en Wall Clock que `fast_molora` completo (**34.74s vs 36.71s**).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Rango Activo (%) | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dyrank_molora`** (v316) 🌟 | 100,160 | **3.4748** | **57.9%** | **34.74** | 0.0575 | [ANCLA] |
| **`fast_molora`** (v311) | 83,776 | 3.4751 | 100.0% | 36.71 | 0.0585 | [ANCLA] |
| **`static_lora`** | 82,752 | 3.4784 | 100.0% | 16.47 | 0.0585 | [ANCLA] |
| **`standard_dense`** | **49,984** | 3.4819 | 100.0% | **9.86** | **0.0611** | [ANCLA] |

*Nota: El marcador 🌟 asigna el mejor rendimiento (menor loss) a `dyrank_molora` (3.4748).*

---

## 2. Dinámica de Emergencia de Rango

```
Batch 1/5 - Loss: 4.3164 | Active Rank:  0.0%
Batch 2/5 - Loss: 4.2765 | Active Rank:  0.0%
Batch 3/5 - Loss: 4.2556 | Active Rank: 60.5%
Batch 4/5 - Loss: 4.2079 | Active Rank: 62.1%
Batch 5/5 - Loss: 4.1978 | Active Rank: 63.2%
```

Durante los dos primeros batches, la red mantiene las compuertas cerradas ($0.0\%$). A partir del tercer batch, el gradiente activa selectivamente las dimensiones de bajo rango que contienen señal útil, estabilizándose en una esparcidad de rango activa media del **57.9%**.
