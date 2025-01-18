from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

plt.rcParams["toolbar"] = "None"


class SpringMassSystem:
    slider_color = "lightgoldenrodyellow"
    default_x_pos = 2.0
    max_points = 500
    title = "Symulacja układu sprężyna-masa"

    def __init__(self):
        self.mass = 1.0
        self.resilience = 10.0
        self.damping = 0.5

        self.current_time = 0
        self.dt = 0.02  # krok czasowy (delta time)
        self.time_modifier = 1.0
        self.x_position = self.default_x_pos
        self.velocity = 0.0

        self.time_data = deque(maxlen=self.max_points)
        self.velocity_data = deque(maxlen=self.max_points)

        self.velocity_plot_active = True

        self.setup_plot()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(10, 6))
        gs = self.fig.add_gridspec(2, 2)
        self.main_plot = self.fig.add_subplot(gs[0])
        self.velocity_plot = self.fig.add_subplot(gs[1])

        self.fig.canvas.manager.set_window_title(self.title)

        self.main_plot.set_xlim(-6, 6)
        self.main_plot.set_ylim(-1, 1)
        self.main_plot.grid(True)
        self.main_plot.set_aspect("equal")

        self.velocity_plot.set_xlabel("Czas [s]")
        self.velocity_plot.set_ylabel("Prędkość [m/s]")
        self.velocity_plot.grid(True)
        (self.velocity_line,) = self.velocity_plot.plot(
            [], [], "b-", lw=1, picker=True
        )
        self.velocity_plot.set_ylim(-10, 10)
        self.velocity_plot.set_xlim(0, 10)

        (self.cursor_point,) = self.velocity_plot.plot([], [], "ro", ms=5)
        self.cursor_text = self.velocity_plot.text(
            0.02, 0.02, "", transform=self.velocity_plot.transAxes
        )

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        (self.spring,) = self.main_plot.plot([], [], "b-", lw=2)
        (self.mass_point,) = self.main_plot.plot([], [], "ro", markersize=20)

        self.setup_sliders()

        self.reset_button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, "Reset")

        # Podłaczenie eventów
        self.time_slider.on_changed(self.update_time_modifier)
        self.mass_slider.on_changed(self.update_mass)
        self.spring_slider.on_changed(self.update_resilience)
        self.damping_slider.on_changed(self.update_damping)
        self.reset_button.on_clicked(self.reset)

    def setup_sliders(self):
        self.time_slider_ax = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.time_slider = Slider(
            self.time_slider_ax,
            "Prędkość symulacji",
            0.1,
            2.0,
            valinit=1.0,
            valstep=0.1,
            color=self.slider_color,
        )

        self.mass_slider_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.mass_slider = Slider(
            self.mass_slider_ax,
            "Masa [kg]",
            0.1,
            5.0,
            valinit=self.mass,
            valstep=0.1,
            color=self.slider_color,
        )

        self.spring_slider_ax = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.spring_slider = Slider(
            self.spring_slider_ax,
            "Sprężystość [N/m]",
            1.0,
            30.0,
            valinit=self.resilience,
            valstep=1.0,
            color=self.slider_color,
        )

        self.damping_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.damping_slider = Slider(
            self.damping_slider_ax,
            "Tłumienie [Ns/m]",
            0.0,
            2.0,
            valinit=self.damping,
            valstep=0.1,
            color=self.slider_color,
        )

    def on_mouse_move(self, event):
        if event.inaxes == self.velocity_plot:
            line_x = list(self.time_data)
            line_y = list(self.velocity_data)

            if not line_x or not line_y:
                self.cursor_point.set_visible(False)
                self.cursor_text.set_visible(False)
                return

            x = event.xdata
            distances = [abs(x_point - x) for x_point in line_x]
            nearest_idx = distances.index(min(distances))

            self.cursor_point.set_data(
                [line_x[nearest_idx]], [line_y[nearest_idx]]
            )
            self.cursor_point.set_visible(True)

            self.cursor_text.set_text(
                f"Czas: {line_x[nearest_idx]:.2f} s\nPrędkość: "
                f"{line_y[nearest_idx]:.2f} m/s"
            )
            self.cursor_text.set_visible(True)

            self.fig.canvas.draw_idle()
        else:
            self.cursor_point.set_visible(False)
            self.cursor_text.set_visible(False)
            self.fig.canvas.draw_idle()

    def create_spring(self, x):
        num_coils = 6
        coil_radius = 0.2
        t = np.linspace(0, 1, 100)

        spring_x = t * x
        spring_y = coil_radius * np.sin(2 * np.pi * num_coils * t)

        return spring_x, spring_y

    def update_mass(self, value):
        self.mass = value

    def update_resilience(self, value):
        self.resilience = value

    def update_damping(self, value):
        self.damping = value

    def update_time_modifier(self, value):
        self.time_modifier = value

    def reset(self, event):
        self.x_position = self.default_x_pos
        self.velocity = 0.0
        self.current_time = 0
        self.time_data.clear()
        self.velocity_data.clear()
        self.velocity_plot.set_xlim(0, 10)
        self.velocity_plot_active = True
        self.velocity_line.set_data([], [])
        self.cursor_point.set_visible(False)
        self.cursor_text.set_visible(False)

    def update(self, frame):
        # Obliczenie przyspieszenia (II zasada Newtona)
        # | https://pl.wikipedia.org/wiki/Zasady_dynamiki_Newtona
        acceleration = (
            -self.resilience * self.x_position - self.damping * self.velocity
        ) / self.mass

        # Całkowanie numeryczne (metoda Eulera) ze skalą czasu
        # | https://pl.wikipedia.org/wiki/Metoda_Eulera
        effective_dt = self.dt * self.time_modifier
        self.velocity += acceleration * effective_dt
        self.x_position += self.velocity * effective_dt
        self.current_time += effective_dt

        if self.velocity_plot_active:
            if self.current_time <= 10:
                self.time_data.append(self.current_time)
                self.velocity_data.append(self.velocity)
                self.velocity_line.set_data(
                    list(self.time_data), list(self.velocity_data)
                )
            else:
                self.velocity_plot_active = False

        spring_x, spring_y = self.create_spring(self.x_position)
        self.spring.set_data(spring_x, spring_y)

        self.mass_point.set_data([self.x_position], [0])

        return (
            self.spring,
            self.mass_point,
            self.velocity_line,
            self.cursor_point,
            self.cursor_text,
        )

    def animate(self):
        _ = FuncAnimation(
            self.fig, self.update, frames=None, interval=20, blit=True
        )
        plt.show()


system = SpringMassSystem()
system.animate()
