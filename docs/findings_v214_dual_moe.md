# V214: El Colapso de los Expertos (Expert Collusion)

## Objetivo
Evaluar si un Enrutador (Router) en una arquitectura de Mixture of Experts (MoE) puede descubrir de forma autónoma que las sumas/restas deben enviarse a un experto Lineal, y las multiplicaciones/divisiones a un experto Logarítmico.

## Resultados del Enrutamiento
| Operación | Experto Lineal | Experto Logarítmico |
| :--- | :--- | :--- |
| Suma (+) | 34.4% | 65.6% |
| Resta (-) | 3.2% | **96.8%** |
| Multiplicación (*) | 47.9% | 52.1% |
| División (/) | 7.0% | **93.0%** |

*Train MSE:* 0.37 | *Estabilidad (OOD):* 7800.0 (Explosión total)

## Conclusiones: ¿Qué salió mal?

¡El experimento arrojó resultados contrarios a la intuición matemática! ¿Por qué el enrutador envió la Resta (que produce números negativos) al Experto Logarítmico (que solo puede escupir números positivos gracias a su `exp()`) en un 96.8% de las veces?

### 1. El Fenómeno de "Expert Collusion" (Colapso MoE)
Como usamos un `Softmax` suave (las salidas se suman ponderadas: $P_{lin} \cdot E_{lin} + P_{log} \cdot E_{log}$), los expertos **no compitieron, sino que hicieron trampa colaborando**.
Si la respuesta correcta era `-3` (Resta), el Experto Log escupía un `+10` gigante, y el Experto Lineal aprendía a escupir `-13` para compensar. La red se convirtió en una maraña espagueti donde ambos expertos dependían vitalmente del otro para anular sus errores mutuos.

### 2. La Explosión OOD
Al depender de esta "resta de errores", en cuanto pasamos al OOD (números más grandes), el Experto Logarítmico explotó (debido al `exp()`), y el Experto Lineal no pudo compensar un número infinito. El error se disparó a 2941.

### 3. La Lección de Arquitectura
Para que un Mixture of Experts desarrolle especialistas reales (como querías en tu hipótesis), el Softmax no puede ser suave. **Debemos forzar Hard-Routing** (ej. Top-1 Gating o Gumbel-Softmax) para que la probabilidad sea `[1.0, 0.0]`. Así, un experto es responsable al 100% del error y no puede pedirle al otro que compense sus deficiencias.
