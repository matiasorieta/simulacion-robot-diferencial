import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
import time
import random


# --- Parámetros de simulación ---
dt = 0.05
T = 60
frames_total = int(T / dt)


# Robot
r_rueda = 0.15
L = 0.5
pulsos_por_rev = 500
ruido_sigma = 0.5
metros_por_pulso = 2 * np.pi * r_rueda / pulsos_por_rev


# Controlador
Kp_theta = 0.5
Kd_theta = 0.6
Ky = 0.3
v_max = 1.5
omega_max = 2


# Variables globales
x_real = y_real = theta_real = 0.0
x_est = y_est = theta_est = 0.0
encoder_left = encoder_right = 0.0


# --- Parámetros del LIDAR económico 2D ---
lidar_interval = int(0.2/ dt)  # Cada 200ms
lidar_noise_pos = 0.05          # Ruido en posición (metros)
lidar_noise_theta = 0.1         # Ruido en orientación (radianes)




# Datos
error_theta_data = []


# Trayectoria deseada
def generar_trayectoria(tipo='circulo'):
    t = np.arange(0, T, dt)
    if tipo == 'ovalo':
        a, b = 3.0, 1.5
        w = 0.3
        x_d = a * np.cos(w * t)
        y_d = b * np.sin(w * t)
    elif tipo == 'circulo':
        r = 3.0
        w = 0.3
        x_d = r * np.cos(w * t)
        y_d = r * np.sin(w * t)
    else:
        raise ValueError("Tipo no soportado")
    
    dx = np.gradient(x_d, dt)
    dy = np.gradient(y_d, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    return t, x_d, y_d, dx, dy, ddx, ddy


# Estado inicial y trayectoria
def reiniciar_estado():
    global t, x_d_vals, y_d_vals, dx_d, dy_d, ddx_d, ddy_d
    global x_real, y_real, theta_real, x_est, y_est, theta_est
    global encoder_left, encoder_right
    global x_hist, y_hist, e_theta_prev
    global error_y_data, time_data, delta_theta_data, delta_pos_data, error_theta_data
    global perturbar_golpe, perturbar_deslizamiento, slide_start_time, slide_duration


    perturbar_golpe = False
    perturbar_deslizamiento = False
    slide_start_time = None
    slide_duration = 0


    t, x_d_vals, y_d_vals, dx_d, dy_d, ddx_d, ddy_d = generar_trayectoria()
    x_real = x_d_vals[0] #+ np.random.uniform(-0.5, 0.5)
    y_real = y_d_vals[0] #+ np.random.uniform(-0.5, 0.5)
    theta_real = np.arctan2(dy_d[0], dx_d[0])


    x_est = x_real
    y_est = y_real
    theta_est = theta_real


    encoder_left = encoder_right = 0
    x_hist = []
    y_hist = []
    e_theta_prev = 0
    error_y_data = []
    time_data = []
    delta_theta_data = []
    delta_pos_data = []
    error_theta_data = []


def lidar_correction():
    global x_est, y_est, theta_est
    # Simula una lectura con ruido desde el LIDAR
    x_lidar = x_real + np.random.normal(0, lidar_noise_pos)
    y_lidar = y_real + np.random.normal(0, lidar_noise_pos)
    theta_lidar = theta_real + np.random.normal(0, lidar_noise_theta)


    # Corrección absoluta (idealizada para este ejemplo)
    x_est = x_lidar
    y_est = y_lidar
    theta_est = theta_lidar




reiniciar_estado()


# --- Gráficos ---
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], figure=fig)


ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])
ax5 = fig.add_subplot(gs[4])


ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal')
ax1.set_title("Seguimiento de trayectoria")
ax1.grid(True)


desired_line, = ax1.plot(x_d_vals, y_d_vals, 'b--', label='Trayectoria deseada')
real_traj_line, = ax1.plot([], [], 'r-', linewidth=2, label='Trayectoria estimada')
robot_icon, = ax1.plot([], [], 'ko', markersize=7, label='Robot')
orientation_line, = ax1.plot([], [], 'k-', linewidth=3)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


ax2.set_xlim(0, T)
ax2.set_ylim(-3, 3)
ax2.set_title("Error de posición (e_y)", fontsize=10)
ax2.grid(True)
ax2.set_xlabel("[s]", labelpad=-1)
ax2.set_ylabel("[m]")
error_line, = ax2.plot([], [], 'g-')


ax5.set_xlim(0, T)
ax5.set_ylim(-2, 2)
ax5.set_title("Perturbación de orientación (Δθ)", fontsize=10)
ax5.grid(True)
ax5.set_xlabel("[s]", labelpad=-1)
ax5.set_ylabel("[rad]")
delta_theta_line, = ax5.plot([], [], 'm-')


ax4.set_xlim(0, T)
ax4.set_ylim(-3, 3)
ax4.set_title("Perturbación de posición (Δx, Δy)", fontsize=10)
ax4.grid(True)
ax4.set_xlabel("[s]", labelpad=-1)
ax4.set_ylabel("[m]")
delta_pos_line, = ax4.plot([], [], 'c-')


ax3.set_xlim(0, T)
ax3.set_ylim(-2, 2)
ax3.set_title("Error de orientación (e_θ)", fontsize=10)
ax3.grid(True)
ax3.set_xlabel("[s]", labelpad=-1)
ax3.set_ylabel("[rad]")
error_theta_line, = ax3.plot([], [], 'orange')


plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95, hspace=1)


# Init
def init():
    real_traj_line.set_data([], [])
    robot_icon.set_data([], [])
    orientation_line.set_data([], [])
    error_line.set_data([], [])
    delta_theta_line.set_data([], [])
    delta_pos_line.set_data([], [])
    error_theta_line.set_data([], [])
    return real_traj_line, robot_icon, orientation_line, error_line, delta_theta_line, delta_pos_line, error_theta_line


# Update
frame_actual = [0]
delta_theta_acumulado = 0
delta_pos_acumulado = 0




def update(frame):
    global x_real, y_real, theta_real, x_est, y_est, theta_est, e_theta_prev
    global encoder_left, encoder_right
    global perturbar_golpe, perturbar_deslizamiento, slide_start_time, slide_duration
    global delta_theta_acumulado, delta_pos_acumulado


    frame = frame_actual[0]
    frame_actual[0] += 1
    if frame_actual[0] >= frames_total:
        frame_actual[0] = 0
        reiniciar_estado()
        init()
        return real_traj_line, robot_icon, orientation_line, error_line, delta_theta_line, delta_pos_line, error_theta_line


    # Trayectoria deseada
    x_d = x_d_vals[frame]
    y_d = y_d_vals[frame]
    dx = dx_d[frame]
    dy = dy_d[frame]
    ddx = ddx_d[frame]
    ddy = ddy_d[frame]


    theta_d = np.arctan2(dy, dx)
    omega_d = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)


    e_y = -np.sin(theta_est) * (x_d - x_est) + np.cos(theta_est) * (y_d - y_est)
    e_theta = np.arctan2(np.sin(theta_d - theta_est), np.cos(theta_d - theta_est))


    v_d = np.sqrt(dx**2 + dy**2)
    
    v = v_d - Ky * e_y
    de_theta = (e_theta - e_theta_prev) / dt
    omega = omega_d + Kp_theta * e_theta + Kd_theta * de_theta + Ky * e_y
    v = np.clip(v, -v_max, v_max)
    omega = np.clip(omega, -omega_max, omega_max)
    e_theta_prev = e_theta


    # --- Movimiento real y perturbaciones ---
    delta_theta = 0
    delta_pos = 0


    # Movimiento teórico basado en control
    v_r = v + (L / 2) * omega
    v_l = v - (L / 2) * omega
    delta_s_r = v_r * dt
    delta_s_l = v_l * dt


    d_center = (delta_s_r + delta_s_l) / 2
    d_theta = (delta_s_r - delta_s_l) / L


    x_next = x_real + d_center * np.cos(theta_real + d_theta / 2)
    y_next = y_real + d_center * np.sin(theta_real + d_theta / 2)
    theta_next = theta_real + d_theta


    # Si hay deslizamiento, modificamos el movimiento real
    if perturbar_deslizamiento:
        elapsed = time.time() - slide_start_time
        if elapsed < slide_duration:
            # Simula que la rueda izquierda patina
            v_r_slide = v + (L / 2) * omega
            v_l_slide = (v - (L / 2) * omega) * 0.3
            delta_s_r = v_r_slide * dt
            delta_s_l = v_l_slide * dt
            d_center_slide = (delta_s_r + delta_s_l) / 2
            d_theta_slide = (delta_s_r - delta_s_l) / L


            x_real += d_center_slide * np.cos(theta_real + d_theta_slide / 2)
            y_real += d_center_slide * np.sin(theta_real + d_theta_slide / 2)
            theta_real += d_theta_slide


            # Perturbación de posición y orientación aplicada en esta perturbación
            delta_theta_acumulado += d_theta_slide
            delta_pos_acumulado += np.sqrt(delta_s_r**2 + delta_s_l**2)
        else:
            perturbar_deslizamiento = False
            # Aplicamos el movimiento normal (post-deslizamiento)
            x_real = x_next
            y_real = y_next
            theta_real = theta_next
            delta_theta_acumulado = 0
            delta_pos_acumulado = 0
    else:
        # Movimiento normal
        x_real = x_next
        y_real = y_next
        theta_real = theta_next
        
        delta_theta_acumulado = 0
        delta_pos_acumulado = 0


    # Golpe aleatorio encima de lo anterior
    if perturbar_golpe:
        dx_random = np.random.uniform(-0.8, 0.8)
        dy_random = np.random.uniform(-0.8, 0.8)
        dtheta_random = np.random.uniform(-0.8, 0.8)
        x_real += dx_random
        y_real += dy_random
        theta_real += dtheta_random
        delta_theta += dtheta_random
        delta_pos += np.sqrt(dx_random**2 + dy_random**2)
        perturbar_golpe = False


    # Odometría estimada (cree que no hubo problema)
    pulsos_r = delta_s_r / metros_por_pulso + np.random.normal(0, ruido_sigma)
    pulsos_l = delta_s_l / metros_por_pulso + np.random.normal(0, ruido_sigma)
    encoder_right += pulsos_r
    encoder_left += pulsos_l


    delta_s_r_recon = pulsos_r * metros_por_pulso
    delta_s_l_recon = pulsos_l * metros_por_pulso
    d_center_est = (delta_s_r_recon + delta_s_l_recon) / 2
    d_theta_est = (delta_s_r_recon - delta_s_l_recon) / L
    x_est += d_center_est * np.cos(theta_est + d_theta_est / 2)
    y_est += d_center_est * np.sin(theta_est + d_theta_est / 2)
    theta_est += d_theta_est


    if delta_theta_acumulado != 0 or delta_pos_acumulado != 0:
        # Actualizar la estimación con las perturbaciones acumuladas
        delta_theta += delta_theta_acumulado
        delta_pos += delta_pos_acumulado


    # Historial
    time_data.append(t[frame])
    error_y_data.append(e_y)
    error_theta_data.append(e_theta)
    delta_theta_data.append(delta_theta)
    delta_pos_data.append(delta_pos)


    # Graficar
    x_hist.append(x_real)
    y_hist.append(y_real)


    real_traj_line.set_data(x_hist, y_hist)
    robot_icon.set_data([x_real], [y_real])
    orientation_line.set_data([x_real, x_real + 0.8 * np.cos(theta_real)],
                              [y_real, y_real + 0.8 * np.sin(theta_real)])


    error_line.set_data(time_data, error_y_data)
    delta_theta_line.set_data(time_data, delta_theta_data)
    delta_pos_line.set_data(time_data, delta_pos_data)
    error_theta_line.set_data(time_data, error_theta_data)


    # Corrección con LIDAR cada cierto intervalo
    if frame % lidar_interval == 0:
        lidar_correction()


    return real_traj_line, robot_icon, orientation_line, error_line, delta_theta_line, delta_pos_line, error_theta_line




# Botones
ax_btn1 = plt.axes([0.1, 0.05, 0.25, 0.06])
ax_btn2 = plt.axes([0.4, 0.05, 0.25, 0.06])
ax_btn3 = plt.axes([0.7, 0.05, 0.2, 0.06])


btn_golpe = Button(ax_btn1, 'Perturbación Golpe')
btn_slide = Button(ax_btn2, 'Deslizamiento')
btn_reset = Button(ax_btn3, 'Reiniciar')


def activar_golpe(event):
    global perturbar_golpe
    perturbar_golpe = True


def activar_deslizamiento(event):
    global perturbar_deslizamiento, slide_start_time, slide_duration
    perturbar_deslizamiento = True
    slide_start_time = time.time()
    slide_duration = random.uniform(1, 2)


def reiniciar(event):
    global frame_actual
    reiniciar_estado()
    frame_actual[0] = 0
    fig.canvas.draw_idle()
    init()


btn_golpe.on_clicked(activar_golpe)
btn_slide.on_clicked(activar_deslizamiento)
btn_reset.on_clicked(reiniciar)


# Animación
ani = FuncAnimation(
    fig,
    update,
    frames=frames_total,
    init_func=init,
    interval=dt * 1000,
    blit=True,
    repeat=True 
)


plt.show()