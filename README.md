﻿# Simulador de Robot Diferencial


Este proyecto implementa un simulador de robot móvil diferencial capaz de seguir trayectorias circulares u ovaladas. Utiliza un controlador PD para la orientación y un controlador proporcional para corregir el error lateral. El simulador incluye sensores virtuales, como encoders y un LIDAR 2D de bajo costo, y permite evaluar el comportamiento del robot ante distintas perturbaciones físicas.

## Requisitos


- Python 3.8 o superior


## Instalación de dependencias


Ejecuta el siguiente comando en la terminal para instalar las dependencias necesarias:


```bash
pip install numpy matplotlib
```


## Ejecución


Desde la terminal, navega a la carpeta del proyecto y ejecuta:


```bash
python simulacion.py
```


## Interfaz e interacciones


La interfaz gráfica incluye los siguientes controles:


- **Perturbación Golpe:** Aplica una fuerza aleatoria que desplaza y rota el robot, permitiendo observar la respuesta del controlador ante perturbaciones externas.
- **Deslizamiento:** Simula el patinamiento de una rueda durante 1 a 2 segundos, afectando la trayectoria del robot.
- **Reiniciar:** Restaura la simulación a su estado inicial.


## Descripción adicional


El simulador es útil para probar algoritmos de control y analizar la respuesta del robot ante condiciones adversas, facilitando la experimentación sin necesidad de hardware físico.
