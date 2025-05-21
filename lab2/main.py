import math

import matplotlib.pyplot as plt

from lab1.main import print_with_breaks
from matrix.matrix import *
from matrix.methods import *


def calculate_task_1(eps: float = 1e-9) -> None:
    print("=== 2.1 ===")

    def f(x):
        return 4**x - 5 * x - 2

    def df(x):
        return math.log(4) * 4**x - 5

    def phi(x):
        return (4**x - 2) / 5

    x0 = 1.0

    print("Метод простой итерации:")
    root_iter, errors_iter, steps_iter = simple_iteration_method(phi, x0, eps)
    print_with_breaks("Корень (итерации)", root_iter)
    print_with_breaks("Погрешности (итерации)", errors_iter)
    print_with_breaks("Число итераций (итерации)", steps_iter)

    plt.plot(
        range(1, len(errors_iter) + 1),
        errors_iter,
        marker="o",
        label="Простая итерация",
    )

    print("\nМетод Ньютона:")
    root_newton, errors_newton, steps_newton = newton_method(f, df, x0, eps)
    print_with_breaks("Корень (Ньютон)", root_newton)
    print_with_breaks("Погрешности (Ньютон)", errors_newton)
    print_with_breaks("Число итераций (Ньютон)", steps_newton)

    plt.plot(
        range(1, len(errors_newton) + 1), errors_newton, marker="s", label="Ньютон"
    )

    plt.xlabel("Номер итерации")
    plt.ylabel("Погрешность")
    plt.yscale("log")
    plt.title("Сходимость методов")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_task_2(eps: float = 1e-9) -> None:
    print("=== 2.2 ===")

    a = 3

    def F(x: Matrix) -> Matrix:
        x1, x2 = x[0][0], x[1][0]
        return Matrix(from_list=[[a * x1 - math.cos(x2)], [a * x2 - math.exp(x1)]])

    def J(x: Matrix) -> Matrix:
        x1, x2 = x[0][0], x[1][0]
        return Matrix(from_list=[[a, math.sin(x2)], [-math.exp(x1), a]])

    def phi(x: Matrix) -> Matrix:
        x1, x2 = x[0][0], x[1][0]
        return Matrix(from_list=[[math.cos(x2) / a], [math.exp(x1) / a]])

    x0 = Matrix(from_list=[[0.5], [0.5]])

    print("Метод простой итерации:")
    root_iter, errors_iter, steps_iter = simple_iteration_system(phi, x0, eps)
    print_with_breaks("Решение (итерация)", root_iter)
    print_with_breaks("Погрешности", errors_iter)
    print_with_breaks("Число итераций", steps_iter)
    plt.plot(
        range(1, steps_iter + 1), errors_iter, label="Простая итерация", marker="o"
    )

    print("\nМетод Ньютона:")
    root_newton, errors_newton, steps_newton = newton_system(F, J, x0, eps)
    print_with_breaks("Решение (Ньютон)", root_newton)
    print_with_breaks("Погрешности", errors_newton)
    print_with_breaks("Число итераций", steps_newton)
    plt.plot(range(1, steps_newton + 1), errors_newton, label="Ньютон", marker="s")

    plt.yscale("log")
    plt.xlabel("Номер итерации")
    plt.ylabel("Погрешность")
    plt.title("Сходимость методов для системы")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    calculate_task_1()
    calculate_task_2()
