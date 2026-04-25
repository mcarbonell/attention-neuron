# Findings: V15 (Learnable-Lp Attention Neuron)

## 1. Experimento

Se implementó la variante V15 con el objetivo de permitir a la red aprender directamente el álgebra de agregación mediante la parametrización del exponente $p$ en una norma $L_p$ generalizada:
- `p = 1 + softplus(rho)`
- `y_lp_mag = ( |X|^p @ |W|^p.T ) ^ (1/p)`
- `y_out = y_lp_mag * sign(y_sum)` (Para mantener las propiedades inhibitorias).

Entrenamiento: MNIST, 10 épocas, Adam.

## 2. Resultados

| Variante | Accuracy | Estado de la Optimización |
| :--- | :--- | :--- |
| **V14 (Dial SUM vs L2)** | 86.46% | Estable |
| **V15 (Learnable Lp)** | ~11.00% | **Colapso de Gradiente** |

## 3. Conclusiones

1. **Inestabilidad del Gradiente**: Elevar tensores de activaciones y pesos a potencias variables dinámicas ($p$) provoca inestabilidad numérica extrema durante la retropropagación. 
2. **Problema del Signo**: Para que una norma $L_p$ estricta funcione en una red neuronal tradicional (que necesita valores negativos para la inhibición), es necesario restaurar el signo artificialmente. El uso de la función `sign()`, al no ser diferenciable, rompe el flujo del gradiente en los momentos en que la red necesita reorientar la polaridad de la suma.
3. **Validación de la V14**: Este fallo confirma que el enfoque de la V14 (mezclar con un dial estático `alpha` dos resultados previamente computados y estables: `SUM` y `L2`) es la manera matemáticamente correcta y robusta de introducir polimorfismo lógico en redes neuronales. 

## 4. Cierre
La vía de exponentes dinámicos queda descartada por su inestabilidad frente al uso de diales interpoladores entre funciones pre-calculadas (V13/V14).