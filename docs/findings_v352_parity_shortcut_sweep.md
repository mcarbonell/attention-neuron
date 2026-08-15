# Findings v352: Falsification Audit of Parity Shortcut & DeltaProduct Real Control

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Refutación de la Hipótesis Par vs. Impar:** El barrido empírico refuta la hipótesis de que los módulos impares no tienen atajo de paridad y muestran mayor ventaja compleja. En $\mathbb{Z}_9$ (impar compuesto $3^2$), `Complex Beta` obtuvo **13.98%** (inferior a `Fixed Beta=2.0` con 14.84%, brecha de **-0.86%**). En cambio, en los módulos pares ($\mathbb{Z}_8$ y $\mathbb{Z}_{12}$), `Complex Beta` sostuvo las mayores ventajas cuantitativas (**+8.36%** y **+6.59%** respectivamente).
- **Eficacia de DeltaProduct Real ($n_h=2$):** En $\mathbb{Z}_3$, la composición de 2 reflexiones reales `DeltaProduct (n_h=2)` alcanzó **48.31% de precisión**, superando a `Complex Beta` (45.10%).

## 1. Resumen Ejecutivo
Se evaluó el barrido de módulos $k \in \{3, 5, 7, 8, 9, 12\}$ bajo 4 brazos comparativos (600 pasos por modelo).

### Tabla de Resultados Brutos del Barrido Módular ($L=64$)
| Módulo $k$ | Tipo de Grupo | Baseline Azar | Real Beta ($\beta \in (0,2)$) | Fixed Real Beta ($\beta \equiv 2.0$) | DeltaProduct Real ($n_h=2$) | Complex Beta ($\beta_t = 1+e^{i\varphi_t}$) | Brecha (Complejo - Isometric) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\mathbb{Z}_3$** | Impar (Primo) | 33.33% | 42.17% | 44.89% | **48.31%** 🌟 | 45.10% | +0.22% | [ANCLA-NEGATIVO] |
| **$\mathbb{Z}_5$** | Impar (Primo) | 20.00% | 26.77% | **44.32%** 🌟 | 26.04% | 28.29% | -16.03% | [ANCLA-NEGATIVO] |
| **$\mathbb{Z}_7$** | Impar (Primo) | 14.29% | 18.89% | 21.16% | 19.72% | **22.59%** | +1.43% | [ANCLA-NEGATIVO] |
| **$\mathbb{Z}_8$** | Par (Compuesto) | 12.50% | 16.14% | 19.22% | 16.47% | **27.58%** 🌟 | **+8.36%** | [ANCLA] |
| **$\mathbb{Z}_9$** | Impar (Compuesto) | 11.11% | **16.31%** 🌟 | 14.84% | 15.09% | 13.98% | **-0.86%** | [ANCLA-NEGATIVO] |
| **$\mathbb{Z}_{12}$** | Par (Compuesto) | 8.33% | 10.22% | 11.34% | 11.13% | **17.93%** 🌟 | **+6.59%** | [ANCLA] |

## 2. Inventario de Arquitectura y Parámetros por Brazo
* **Dimensiones:** $d_{\text{model}} = 64$, $n_{\text{layers}} = 4$, $n_{\text{heads}} = 4$, $d_k = 16$.
* **Conteo Comparativo:**
  - `Real Beta`: **$200,343$ parámetros** ($1.000\times$).
  - `Fixed Real Beta=2.0`: **$199,303$ parámetros** ($0.995\times$).
  - `DeltaProduct Real (n_h=2)`: **$218,023$ parámetros** ($1.088\times$, $+8.8\%$ parámetros por doble proyección de clave y compuerta).
  - `Complex Beta`: **$200,343$ parámetros** ($1.000\times$, exactamente iso-paramétrico con Real Beta).

## 3. Diagnóstico Causal
1. **Refutación Teórica:** La idea de que los módulos pares atenúan la ganancia compleja queda desmentida por los datos: $\mathbb{Z}_8$ (+8.36%) y $\mathbb{Z}_{12}$ (+6.59%) presentan las mayores brechas relativas a favor de `Complex Beta`.
2. **Desempeño de DeltaProduct:** La composición de dos Householders reales por token alcanza la máxima precisión en grupos pequeños ($\mathbb{Z}_3$), demostrando que 2 reflexiones reales pueden generar rotaciones efectivas en planos acotados, si bien requiere un $+8.8\%$ más de parámetros entrenables.

## 4. Conclusión
La hipótesis de "par vs. impar" como divisor de expresividad no se sostiene empíricamente. Las ventajas de la parametrización compleja deben evaluarse por su capacidad de rotación continua en $U(d)$ e isometría garantizada, sin asumir patrones simplificados por paridad del módulo.

