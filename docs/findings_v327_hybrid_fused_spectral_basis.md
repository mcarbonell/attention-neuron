# Hallazgos Experimento v327: Fusión Tri-Espectral Híbrida (Fase 6)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** Se asumía que las bases espectrales debían usarse de forma aislada (solo FWHT o solo DCT-II) y que promediar bases introduciría interferencia entre dominios.
* **Resultado Certificado del Experimento v327 [ANCLA]:** **VICTORIA ROTUNDA DE LA FUSIÓN TRI-ESPECTRAL.**
  1. **Fused Tri-Spectral es la Ganadora Absoluta (0.0155 Loss, 99.76% Acc):** Al combinar las tres transformadas ortogonales (FWHT + DCT-II + DWT Haar) en paralelo en el FFN, el modelo superó a las 3 bases puras aisladas a 15 épocas.
  2. **Mayor Eficiencia Paramétrica (PEI: 10.8671):** Alcanza un PEI superior al de DCT-II pura (9.16) y FWHT pura (8.23).
  3. **Identificación de la Siguiente Mejora (Sustrate Lerp Gated):** El promedio simple de las 3 ramas ($1/3$ cada una) funciona excelentemente, pero puede optimizarse sustituyendo el promedio rígido por una combinación convexa aprendible (*Lerp*) por canal/capa con informe de sustratos elegidos (`v328`).

---

## 1. Tabla de Resultados Empíricos (15 Épocas)

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64, d=128$, 5 capas espectrales profundas, 15 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo Transformer | Sustrato Espectral | Parámetros | Loss Final (15 Épocas) | Accuracy % | Wall Clock (s) | PEI | Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fused Tri-Spectral`** 🌟 | FWHT + DCT-II + Haar | 854,144 | **0.0155** | **99.76%** | 369.72 | **10.8671** | [ANCLA] |
| **`Pure DCT-II`** | Cosenos Reales Armónicos | 685,184 | 0.0187 | 99.70% | 270.12 | 9.1632 | [ANCLA] |
| **`Pure FWHT`** | Binario Discreto $\pm 1$ | 685,184 | 0.0208 | 99.61% | **249.52** | 8.2306 | [ANCLA] |
| **`Pure DWT Haar`** | Ondículas Multi-Resolución | 685,184 | 0.0611 | 98.86% | 261.21 | 2.8025 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss y mayor PEI a `Fused Tri-Spectral`.*

---

## 2. Recomendación Técnica

La combinación de transformadas demuestra que diferentes rasgos se benefician de diferentes geometrías espectrales. El paso a seguir en `v328` es implementar **Aprendizaje de Selección de Sustrato mediante Lerp/Softmax Router** para generar un reporte automático del porcentaje de cada sustrato sintonizado por la red.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La fusión no está iso-capacidad frente a las bases puras: usa tres ramas y una proyección `combine` de entrada $6d$, mientras las puras usan cuatro bancos y $4d$. La mejora puede venir de esa capacidad de combinación y del schedule, no de complementariedad de bases. Faltan validación, semillas y control con base ortogonal aleatoria. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
