import math

import matplotlib.pyplot as plt

from lab1.main import print_with_breaks
from matrix.matrix import *
from matrix.methods import *


def calculate_task_1(eps: float = 1e-9) -> None:
    print('=== 2.1 ===')

    def f(x):
        return 4 ** x - 5 * x - 2

    def df(x):
        return math.log(4) * 4 ** x - 5

    # phi(x) выбрана как приближённая итерационная функция
    # Решаем f(x) = 0 <=> x = phi(x) = (4^x - 2) / 5
    def phi(x):
        return (4 ** x - 2) / 5

    x0 = 1.0

    print("Метод простой итерации:")
    try:
        root_iter, errors_iter, steps_iter = simple_iteration_method(phi, x0, eps)
        print_with_breaks("Корень (итерации)", root_iter)
        print_with_breaks("Погрешности (итерации)", errors_iter)
        print_with_breaks("Число итераций (итерации)", steps_iter)

        # График
        plt.plot(range(1, len(errors_iter) + 1), errors_iter, marker='o', label='Простая итерация')
    except Exception as e:
        print("Ошибка в методе простой итерации:", e)

    print("\nМетод Ньютона:")
    try:
        root_newton, errors_newton, steps_newton = newton_method(f, df, x0, eps)
        print_with_breaks("Корень (Ньютон)", root_newton)
        print_with_breaks("Погрешности (Ньютон)", errors_newton)
        print_with_breaks("Число итераций (Ньютон)", steps_newton)

        plt.plot(range(1, len(errors_newton) + 1), errors_newton, marker='s', label='Ньютон')
    except Exception as e:
        print("Ошибка в методе Ньютона:", e)

    # Отображение графика
    plt.xlabel('Номер итерации')
    plt.ylabel('Погрешность')
    plt.yscale('log')  # логарифмическая шкала по Y
    plt.title('Сходимость методов')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calculate_task_1(eps=1e-9)
