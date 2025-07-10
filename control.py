# Re-ejecutar tras el reinicio del estado


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- Parámetros de simulación ---
dt = 0.05
T = 100
t = np.arange(0, T, dt)


# Parámetros físicos del robot
r_rueda = 0.05  # radio de rueda (m)
L = 0.2         # distancia entre ruedas (m)
pulsos_por_rev = 500
ruido_sigma = 0.5  # desviación estándar del ruido en pulsos


# Conversión: metros a pulsos
metros_por_pulso = 2 * np.pi * r_rueda / pulsos_por_rev


# Trayectoria deseada
def generar_trayectoria():
    r = 2.0
    w = 0.3
    x_d = r * np.cos(w * t)
    y_d = r * np.sin(w * t)
    dx = -r * w * np.sin(w * t)
    dy =  r * w * np.cos(w * t)
    ddx = -r * w**2 * np.cos(w * t)
    ddy = -r * w**2 * np.sin(w * t)
    return x_d, y_d, dx, dy, ddx, ddy


x_d_vals, y_d_vals, dx_d, dy_d, ddx_d, ddy_d = generar_trayectoria()


# Estado inicial
x, y, theta = x_d_vals[0], y_d_vals[0], np.arctan2(dy_d[0], dx_d[0])
# Empieza en una posición randomizada cerca del inicio de la trayectoria
x = x + np.random.uniform(-0.5, 0.5)
y = y + np.random.uniform(-0.5, 0.5)
encoder_left = 0
encoder_right = 0


# Historiales
x_hist = [x]
y_hist = [y]


# Controlador
Kp_theta = 3.2
Kd_theta = 0.8
Ky = 2
e_theta_prev = 0.0


# Límites
v_max = 1.5
omega_max = 2


# --- Gráficos ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_aspect('equal')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.grid(True)
ax1.set_title("Seguimiento de trayectoria - Robot diferencial")


desired_line, = ax1.plot(x_d_vals, y_d_vals, 'b--', linewidth=2.5, label='Trayectoria deseada')
real_traj_line, = ax1.plot([], [], 'r-', linewidth=2.5, label='Trayectoria real')
robot_icon, = ax1.plot([], [], 'ko', markersize=8, label='Robot')
orientation_line, = ax1.plot([], [], 'k-', linewidth=2.5)
ax1.legend()


# Zonas de error
thresholds = {
    "±10% (Normalidad)": (0.10, 'green'),
    "±15% (Error bajo)": (0.15, 'orange'),
    "±20% (Error medio)": (0.20, 'orangered'),
    "±25% (Error alto)": (0.25, 'red'),
}
for label, (th, color) in thresholds.items():
    ax2.axhline(th, color=color, linestyle='--', linewidth=1.2, label=label)
    ax2.axhline(-th, color=color, linestyle='--', linewidth=1.2)


ax2.grid(True)
ax2.set_xlim(0, T)
ax2.set_ylim(-0.6, 0.6)
ax2.set_title("Error lateral (e_y)")
error_y_line, = ax2.plot([], [], 'g-', label='Error lateral')
ax2.legend(loc='upper right', fontsize=8)


error_y_data = []
time_data = []


def init():
    real_traj_line.set_data([x], [y])
    robot_icon.set_data([x], [y])
    orientation_line.set_data([x, x + 0.4 * np.cos(theta)],
                              [y, y + 0.4 * np.sin(theta)])
    error_y_line.set_data([], [])
    return real_traj_line, robot_icon, orientation_line, error_y_line


def update(frame):
    global x, y, theta, e_theta_prev, encoder_left, encoder_right


    # Trayectoria deseada en t
    x_d = x_d_vals[frame]
    y_d = y_d_vals[frame]
    dx = dx_d[frame]
    dy = dy_d[frame]
    ddx = ddx_d[frame]
    ddy = ddy_d[frame]


    theta_d = np.arctan2(dy, dx)
    omega_d = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)


    # Error geométrico
    e_y = -np.sin(theta) * (x_d - x) + np.cos(theta) * (y_d - y)
    e_theta = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))


    v_d = np.sqrt(dx**2 + dy**2)
    v = v_d - Ky * e_y
    de_theta = (e_theta - e_theta_prev) / dt
    omega = omega_d + Kp_theta * e_theta + Kd_theta * de_theta


    v = np.clip(v, -v_max, v_max)
    omega = np.clip(omega, -omega_max, omega_max)


    e_theta_prev = e_theta


    # Cinemática diferencial
    v_r = v + (L / 2) * omega
    v_l = v - (L / 2) * omega


    # Pulsos de encoder simulados (con ruido)
    delta_s_r = v_r * dt
    delta_s_l = v_l * dt


    pulsos_r = delta_s_r / metros_por_pulso + np.random.normal(0, ruido_sigma)
    pulsos_l = delta_s_l / metros_por_pulso + np.random.normal(0, ruido_sigma)


    encoder_right += pulsos_r
    encoder_left += pulsos_l


    # Distancias reconstruidas desde los pulsos
    delta_s_r_recon = pulsos_r * metros_por_pulso
    delta_s_l_recon = pulsos_l * metros_por_pulso


    # Estimación de movimiento
    d_center = (delta_s_r_recon + delta_s_l_recon) / 2
    d_theta = (delta_s_r_recon - delta_s_l_recon) / L


    x += d_center * np.cos(theta + d_theta / 2)
    y += d_center * np.sin(theta + d_theta / 2)
    theta += d_theta


    # Historial
    x_hist.append(x)
    y_hist.append(y)


    # Actualizar gráficas
    real_traj_line.set_data(x_hist, y_hist)
    robot_icon.set_data([x], [y])
    orientation_line.set_data([x, x + 0.4 * np.cos(theta)],
                              [y, y + 0.4 * np.sin(theta)])
    error_y_data.append(e_y)
    time_data.append(frame * dt)
    error_y_line.set_data(time_data, error_y_data)


    return real_traj_line, robot_icon, orientation_line, error_y_line


ani = FuncAnimation(fig, update, frames=len(t),
                    init_func=init, blit=True, interval=dt * 1000)
plt.show()
