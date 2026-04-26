# The Soft Window (V30): Atención Topológica Eficiente

**Documento de Teoría y Optimización de Hardware**
*Evolución de la V28/V29 (Gaussian Splatting)*

---

## 1. El Problema del Gaussian Splatting en 2D
La arquitectura de "Atención Continua" (V28/V29) demostró que podemos eliminar las matrices de pesos discretas ($W$) y sustituirlas por óvalos gaussianos paramétricos ($c_x, c_y, \sigma_x, \sigma_y, \rho, A$). 

**El Cuello de Botella Computacional:**
Aunque el número de parámetros entrenables cae un 99.9% (solo 6 números por detector), el coste computacional ($FLOPs$) se dispara. Para generar la máscara del óvalo en una imagen de resolución $H \times W$, la GPU debe evaluar la función exponencial de la distancia de Mahalanobis en cada píxel del grid 2D: Coste $O(H \times W)$.

Para imágenes de alta resolución (ej. 4K), este cálculo denso bloquea los núcleos CUDA, anulando la ventaja de no tener que leer grandes matrices de memoria (VRAM).

## 2. La Solución: "The Soft Window" (El Rectángulo Diferenciable)
Para que la Atención Topológica sea viable en hardware moderno y mantenga la diferenciabilidad perfecta, proponemos sustituir el óvalo 2D por un **Rectángulo Suave Separable**.

En lugar de evaluar una función 2D compleja, evaluamos dos funciones 1D independientes (una para el eje X y otra para el eje Y) y calculamos su producto exterior.

### 2.1. Matemáticas del Rectángulo Suave
Cada detector (neurona) aprende solo 4 parámetros espaciales que definen las coordenadas de la "ventana":
- $X_{min}, X_{max}$ (Bordes horizontales)
- $Y_{min}, Y_{max}$ (Bordes verticales)

Para que el gradiente fluya y la red pueda "mover" los bordes, usamos la función Sigmoide ($\sigma$) con un factor de temperatura $T$ que controla la nitidez del borde.

**Máscara en el Eje X (Vector 1D):**
$$ M_x(x) = \sigma(T \cdot (x - X_{min})) - \sigma(T \cdot (x - X_{max})) $$

**Máscara en el Eje Y (Vector 1D):**
$$ M_y(y) = \sigma(T \cdot (y - Y_{min})) - \sigma(T \cdot (y - Y_{max})) $$

**Máscara 2D Final (La Ventana):**
$$ M_{2D}(x, y) = M_x(x) \otimes M_y(y) $$

*(El producto exterior $\otimes$ crea la matriz 2D recortando el área de intersección).*

## 3. Ventajas de la "Soft Window"

1.  **Velocidad Extrema (Separabilidad)**: El cálculo pasa de ser $O(H \times W)$ a ser **$O(H + W)$**. Para una imagen 4K ($3840 \times 2160$), evaluar el óvalo Gaussiano requería $\sim 8.3$ millones de operaciones complejas. La *Soft Window* solo requiere evaluar $3840$ sigmoides en X, $2160$ sigmoides en Y, y hacer un broadcasting rápido. ¡Una reducción computacional masiva!
2.  **Ultra-Eficiencia Paramétrica**: Solo **4 parámetros espaciales** por detector (más la amplitud $A$). Es incluso más barato que los 6 parámetros del Gaussiano.
3.  **Hardware Friendly**: Las GPUs modernas (Tensor Cores) están diseñadas para hacer el producto exterior de dos vectores de forma nativa e instantánea.
4.  **Invariancia de Traslación Perfecta**: La neurona aprende un "Crop Diferenciable" (Soft Crop). Si el objeto de interés se mueve a la esquina, el gradiente simplemente empuja $X_{min}$ y $Y_{min}$ hacia esa coordenada sin modificar el "conocimiento" interno de la red.

## 4. Próximos Pasos (Implementación V30)
- Desarrollar la capa `SoftWindow2D` en PyTorch usando broadcasting explícito (`M_x.unsqueeze(2) * M_y.unsqueeze(1)`).
- Validar que el gradiente fluye correctamente a través de los límites ($X_{min}, X_{max}$) empujando la ventana hacia objetos relevantes en CIFAR-10.
- Comparar el *Wall Clock Time* (tiempo real de ejecución) entre la V28 (Gaussian) y la futura V30 (Soft Window) para confirmar el speedup asintótico.
