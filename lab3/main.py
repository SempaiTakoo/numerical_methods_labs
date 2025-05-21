import math

import matplotlib.pyplot as plt

from lab1.main import print_with_breaks
from matrix.matrix import *
from matrix.methods import *


def calculate_task_1(eps: float = 1e-9) -> None:
    print("=== 3.1 ===")

    def f(x):
        return math.exp(x) + x

    x_star = -0.5

    def run_case(X):
        Y = [f(x) for x in X]
        lagrange = lagrange_interpolation(X, Y, x_star)
        newton = newton_interpolation(X, Y, x_star)
        exact = f(x_star)
        print_with_breaks(f'X = {X}', '')
        print_with_breaks("Y", Y)
        print_with_breaks("Интерполяция Лагранжа", lagrange)
        print_with_breaks("Интерполяция Ньютона", newton)
        print_with_breaks("Точное значение", exact)
        print_with_breaks("Ошибка Лагранжа", abs(lagrange - exact))
        print_with_breaks("Ошибка Ньютона", abs(newton - exact))
        print('-' * 40)

        xs = [X[0] + i * (X[-1] - X[0]) / 200 for i in range(201)]
        ys_f = [f(x) for x in xs]
        ys_lagrange = [lagrange_interpolation(X, Y, x) for x in xs]
        ys_newton = [newton_interpolation(X, Y, x) for x in xs]

        plt.figure()
        plt.plot(xs, ys_f, label='f(x)', linewidth=2)
        plt.plot(xs, ys_lagrange, '--', label='Lagrange', linewidth=2)
        plt.plot(xs, ys_newton, ':', label='Newton', linewidth=2)
        plt.scatter(X, Y, color='red', label='Узлы')
        plt.scatter([x_star], [exact], color='green', label='x*')
        plt.title("Интерполяция Лагранжа и Ньютона")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Интерполяция при X = [-2, -1, 0, 1]:")
    run_case([-2, -1, 0, 1])

    print("Интерполяция при X = [-2, -1, 0.2, 1]:")
    run_case([-2, -1, 0.2, 1])


def calculate_task_2():
    print("=== 3.2 ===")

    def f(x):
        return math.exp(x) + x

    X = [-2.0, -1.0, 0.0, 1.0, 2.0]
    Y = [-1.8647, -0.63212, 1.0, 3.7183, 9.3891]
    x_star = -0.5

    result = cubic_spline_interpolation(X, Y, x_star)

    print_with_breaks("Кубический сплайн в x* = -0.5", result)

    xs = [X[0] + i * (X[-1] - X[0]) / 200 for i in range(201)]
    ys_spline = [cubic_spline_interpolation(X, Y, x) for x in xs]
    ys_f = [f(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys_f, label='f(x)', linewidth=2)
    plt.plot(xs, ys_spline, '--', label='Spline', linewidth=2)
    plt.scatter(X, Y, color='red', label='Узлы')
    plt.scatter([x_star], [result], color='green', label='x*')
    plt.title("Кубический сплайн")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_task_3():
    X = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    Y = [-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891]

    coeffs_1 = least_squares_polynomial(X, Y, degree=1)
    Y_pred_1 = [evaluate_polynomial(coeffs_1, x) for x in X]
    error_1 = sum((Y[i] - Y_pred_1[i]) ** 2 for i in range(len(X)))

    coeffs_2 = least_squares_polynomial(X, Y, degree=2)
    Y_pred_2 = [evaluate_polynomial(coeffs_2, x) for x in X]
    error_2 = sum((Y[i] - Y_pred_2[i]) ** 2 for i in range(len(X)))

    print_with_breaks("Коэффициенты линейного МНК", coeffs_1)
    print_with_breaks("Сумма квадратов ошибок (1 степень)", error_1)

    print_with_breaks("Коэффициенты квадратичного МНК", coeffs_2)
    print_with_breaks("Сумма квадратов ошибок (2 степень)", error_2)

    x_plot = [x / 10 for x in range(-30, 21)]
    y_func = [math.exp(x) + x for x in x_plot]
    y_line = [evaluate_polynomial(coeffs_1, x) for x in x_plot]
    y_quad = [evaluate_polynomial(coeffs_2, x) for x in x_plot]

    plt.plot(X, Y, 'ko', label='Табличные данные')
    plt.plot(x_plot, y_func, 'g--', label='Оригинальная функция f(x) = e^x + x')
    plt.plot(x_plot, y_line, 'b-', label='МНК: линейный')
    plt.plot(x_plot, y_quad, 'r-', label='МНК: квадратичный')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Аппроксимация МНК")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calculate_task_1()
    calculate_task_2()
    calculate_task_3()
