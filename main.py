import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


class OptimizedRobotArm:
    def __init__(self, root):
        self.root = root
        self.root.title("Оптимизированный 3D манипулятор")

        # Параметры манипулятора
        self.base_height = 1.0
        self.arm_length = 0.8
        self.forearm_length = 0.6
        self.base_rotation = 0
        self.arm_angle = 45
        self.forearm_angle = 30
        self.target = None
        self.epsilon = 1e-5  # Точность оптимизации
        self.animation_steps = 20  # Количество шагов анимации

        # Создаем интерфейс
        self.create_controls()
        self.create_3d_view()
        self.update_arm()

    def create_controls(self):
        """Панель управления"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Ползунки управления
        tk.Label(control_frame, text="Вращение основания (0-360°):").pack()
        self.rot_slider = tk.Scale(control_frame, from_=0, to=360, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_angle('rotation', v))
        self.rot_slider.pack()

        tk.Label(control_frame, text="Угол подъёма руки (0-90°):").pack()
        self.arm_slider = tk.Scale(control_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_angle('arm', v))
        self.arm_slider.set(45)
        self.arm_slider.pack()

        tk.Label(control_frame, text="Угол предплечья (0-90°):").pack()
        self.forearm_slider = tk.Scale(control_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                                       command=lambda v: self.update_angle('forearm', v))
        self.forearm_slider.set(30)
        self.forearm_slider.pack()

        # Поля для ввода цели
        tk.Label(control_frame, text="Целевая точка (x,y,z):").pack(pady=(10, 0))
        self.target_x = tk.Entry(control_frame)
        self.target_x.pack()
        self.target_y = tk.Entry(control_frame)
        self.target_y.pack()
        self.target_z = tk.Entry(control_frame)
        self.target_z.pack()

        # Кнопки управления
        tk.Button(control_frame, text="Задать цель", command=self.set_target).pack(pady=5)
        tk.Button(control_frame, text="Следовать", command=self.follow_target).pack(pady=5)
        tk.Button(control_frame, text="Сброс", command=self.reset).pack()

    def create_3d_view(self):
        """3D визуализация"""
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Фиксированные границы пространства
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Оптимизированный 3D манипулятор')

        # Элементы манипулятора
        self.base = self.ax.plot([], [], [], 'k-', linewidth=6)[0]
        self.arm = self.ax.plot([], [], [], 'b-', linewidth=4)[0]
        self.forearm = self.ax.plot([], [], [], 'r-', linewidth=2)[0]
        self.end_effector = self.ax.plot([], [], [], 'go', markersize=8)[0]
        self.target_point = self.ax.plot([], [], [], 'ro', markersize=8, alpha=0.5)[0]

        # Рабочая область
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 1.4 * np.cos(theta)
        y = 1.4 * np.sin(theta)
        self.workspace = self.ax.plot(x, y, np.zeros_like(x), 'g--', alpha=0.3)[0]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def calculate_positions(self):
        """Вычисление координат"""
        rot_rad = np.radians(self.base_rotation)
        arm_rad = np.radians(self.arm_angle)
        forearm_rad = np.radians(self.forearm_angle)

        # Базовые точки
        base_bottom = (0, 0, 0)
        base_top = (0, 0, self.base_height)

        # Плечо
        elbow_x = self.arm_length * np.cos(rot_rad) * np.sin(arm_rad)
        elbow_y = self.arm_length * np.sin(rot_rad) * np.sin(arm_rad)
        elbow_z = self.base_height + self.arm_length * np.cos(arm_rad)

        # Предплечье
        end_x = elbow_x + self.forearm_length * np.cos(rot_rad) * np.sin(arm_rad + forearm_rad)
        end_y = elbow_y + self.forearm_length * np.sin(rot_rad) * np.sin(arm_rad + forearm_rad)
        end_z = elbow_z + self.forearm_length * np.cos(arm_rad + forearm_rad)

        return base_bottom, base_top, (elbow_x, elbow_y, elbow_z), (end_x, end_y, end_z)

    def update_angle(self, angle_type, value):
        """Обновление углов"""
        if angle_type == 'rotation':
            self.base_rotation = float(value)
        elif angle_type == 'arm':
            self.arm_angle = float(value)
        elif angle_type == 'forearm':
            self.forearm_angle = float(value)

        self.update_arm()

    def update_arm(self):
        """Обновление отображения"""
        base_bottom, base_top, elbow, end = self.calculate_positions()

        # Обновляем линии
        self.base.set_data([base_bottom[0], base_top[0]], [base_bottom[1], base_top[1]])
        self.base.set_3d_properties([base_bottom[2], base_top[2]])

        self.arm.set_data([base_top[0], elbow[0]], [base_top[1], elbow[1]])
        self.arm.set_3d_properties([base_top[2], elbow[2]])

        self.forearm.set_data([elbow[0], end[0]], [elbow[1], end[1]])
        self.forearm.set_3d_properties([elbow[2], end[2]])

        self.end_effector.set_data([end[0]], [end[1]])
        self.end_effector.set_3d_properties([end[2]])

        # Обновляем цель
        if self.target:
            self.target_point.set_data([self.target[0]], [self.target[1]])
            self.target_point.set_3d_properties([self.target[2]])

        self.canvas.draw()

    def set_target(self):
        """Установка целевой точки"""
        try:
            x = float(self.target_x.get())
            y = float(self.target_y.get())
            z = float(self.target_z.get())

            # Проверка достижимости
            distance = np.sqrt(x ** 2 + y ** 2 + (z - self.base_height) ** 2)
            max_reach = self.arm_length + self.forearm_length

            if distance > max_reach:
                print(f"Цель слишком далеко (макс. {max_reach:.1f} единиц)")
                return

            self.target = (x, y, z)
            self.update_arm()

        except ValueError:
            print("Ошибка ввода координат")
            self.target = None
            self.update_arm()

    def inverse_kinematics(self, target):
        """Точная обратная кинематика с оптимизацией"""

        def error_function(angles):
            arm, forearm = angles

            # Вычисляем положение конечной точки
            x_pos = (self.arm_length * np.sin(arm) +
                     self.forearm_length * np.sin(arm + forearm))
            z_pos = (self.base_height +
                     self.arm_length * np.cos(arm) +
                     self.forearm_length * np.cos(arm + forearm))

            # Расстояние до цели в плоскости XY
            xy_distance = np.sqrt(target[0] ** 2 + target[1] ** 2)

            # Общая ошибка
            error = (xy_distance - x_pos) ** 2 + (target[2] - z_pos) ** 2
            return error

        # Начальные углы (в радианах)
        initial_angles = [np.radians(self.arm_angle), np.radians(self.forearm_angle)]

        # Ограничения углов
        bounds = [(0, np.pi / 2), (0, np.pi / 2)]

        # Оптимизация с высокой точностью
        options = {'ftol': self.epsilon, 'maxiter': 100}
        result = minimize(error_function, initial_angles,
                          bounds=bounds, method='L-BFGS-B', options=options)

        if not result.success:
            print("Оптимизация не удалась:", result.message)
            return self.base_rotation, self.arm_angle, self.forearm_angle

        optimal_arm, optimal_forearm = result.x
        optimal_rotation = np.degrees(np.arctan2(target[1], target[0]))

        return optimal_rotation, np.degrees(optimal_arm), np.degrees(optimal_forearm)

    def follow_target(self):
        """Плавное движение к цели с анимацией"""
        if not self.target:
            print("Цель не задана!")
            return

        try:
            # Сначала получаем точные целевые углы
            target_rot, target_arm, target_forearm = self.inverse_kinematics(self.target)

            # Вычисляем шаги для анимации
            rot_steps = np.linspace(self.base_rotation, target_rot, self.animation_steps)
            arm_steps = np.linspace(self.arm_angle, target_arm, self.animation_steps)
            forearm_steps = np.linspace(self.forearm_angle, target_forearm, self.animation_steps)

            # Анимация движения
            for i in range(self.animation_steps):
                self.base_rotation = rot_steps[i]
                self.arm_angle = arm_steps[i]
                self.forearm_angle = forearm_steps[i]

                # Обновляем слайдеры
                self.rot_slider.set(self.base_rotation)
                self.arm_slider.set(self.arm_angle)
                self.forearm_slider.set(self.forearm_angle)

                self.update_arm()
                self.root.update()
                self.root.after(30)  # Задержка для плавности анимации

            # Финальная точная установка
            self.base_rotation = target_rot
            self.arm_angle = target_arm
            self.forearm_angle = target_forearm
            self.update_arm()

        except Exception as e:
            print(f"Ошибка при движении к цели: {e}")

    def reset(self):
        """Сброс в начальное положение"""
        self.rot_slider.set(0)
        self.arm_slider.set(45)
        self.forearm_slider.set(30)
        self.target_x.delete(0, tk.END)
        self.target_y.delete(0, tk.END)
        self.target_z.delete(0, tk.END)
        self.target = None
        self.update_arm()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizedRobotArm(root)
    root.mainloop()