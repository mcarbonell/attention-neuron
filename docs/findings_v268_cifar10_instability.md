# Findings v268: CIFAR-10 PID Instability

## Observación
La configuración (1, 100, 10) que dominó en MNIST fracasó en CIFAR-10 (-0.59% vs Adam).

## Análisis del Fallo
- **Instabilidad**: Se observaron oscilaciones violentas en el Accuracy (73% -> 69% -> 72%).
- **Causa**: La ganancia integral (Ki=100) es demasiado alta para datos ruidosos. El optimizador acumula "inercia" en direcciones de gradiente obsoletas.
- **Conclusión**: El "Industrial PID" requiere una sintonización dependiente de la relación señal/ruido (SNR) de la tarea.
