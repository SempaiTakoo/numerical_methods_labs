from __future__ import annotations
import math


class Matrix:

    def __init__(
        self, rows: int = 0, cols: int = 0,
        is_indentity: bool = False,
        from_list: None | list[list[float]] = None,
        from_filepath: str = None
    ) -> None:
        if from_list:
            self.__matrix = from_list
            self.__rows, self.__cols = len(from_list), len(from_list[0])
            return

        if from_filepath:
            self.read_file(path=from_filepath)
            return

        if is_indentity and rows != cols:
            raise ValueError('Единичная матрица должна быть квадратной.')

        self.__rows, self.__cols = rows, cols
        self.__matrix: list[list[float]] = [
            [
                1 if is_indentity and i == j else 0
                for j in range(self.__cols)
            ]
            for i in range(self.__rows)
        ]

    def rows_count(self) -> int:
        return self.__rows

    def cols_count(self) -> int:
        return self.__cols

    def rows_cols_count(self) -> tuple[int, int]:
        return self.__rows, self.__cols

    def is_vector(self) -> bool:
        return self.rows_cols_count()[1] == 1

    def read_file(self, path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                self.__rows, self.__cols = map(int, file.readline().split(' '))
                self.__matrix = [
                    list(map(float, file.readline().split(' ')))
                    for _ in range(self.__rows)
                ]

                if self.__rows and self.__cols != len(self.__matrix[0]):
                    raise ValueError
        except FileNotFoundError:
            print(f'Файл {path} не найден.')
        except ValueError:
            print(f'Файл содержит некорректные данные.')
        except Exception as e:
            print(f'Произошла ошибка: {e}')

    def __str__(self) -> str:
        max_num_width = max(
            len(str(round(self.__matrix[i][j], 2)))
            for i in range(self.__rows)
            for j in range(self.__cols)
        )
        return '\n'.join(
            ' '.join(
                str(round(self.__matrix[i][j], 2)).rjust(max_num_width)
                for j in range(self.__cols)
            )
            for i in range(self.__rows)
        )

    def get_column(self, index: int) -> Matrix:
        return Matrix(from_list=[
            [self[row][index]] for row in range(self.__rows)
        ])

    def __getitem__(self, index: int | tuple[int, int]) -> float:
        if isinstance(index, int):
            return self.__matrix[index]

        if isinstance(index, tuple):
            i, j = index
            return self.__matrix[i][j]

        raise TypeError('Индексы матрицы должны быть int или tuple[int, int].')

    def __setitem__(self, index: int | tuple[int, int], value: float) -> None:
        if isinstance(index, int):
            if isinstance(value, float) and self.is_vector():
                self.__matrix[index][0] = value
                return
            if isinstance(value, Matrix) and value.is_vector():
                for row in range(self.rows_count()):
                    self[row][index] = value[row][0]
                return

        if (
            isinstance(index, tuple)
            and all(isinstance(elem, int) for elem in index)
        ):
            i, j = index
            self.__matrix[i][j] = value

        raise TypeError(
            f'Не предусмотрено поведение для типа данных {type(index)}.'
        )

    def __mul__(self, other: int | float | Matrix) -> Matrix:
        n, m = self.rows_cols_count()
        if isinstance(other, (int, float)):
            result = Matrix(n, m)
            for i in range(n):
                for j in range(m):
                    result[i][j] = other * self[i][j]
            return result
        if isinstance(other, Matrix):
            p, q = other.rows_cols_count()
            if m != p:
                raise ValueError(
                    'Матрицы должны быть совместимы, чтобы быть перемноженными.'
                )

            result = Matrix(n, q)
            for i in range(n):
                for j in range(q):
                    for k in range(m):
                        result[i][j] += self[i][k] * other[k][j]

            return result

    def minor(self, row: int, column: int) -> Matrix:
        minor = Matrix(from_list=[
            [self.__matrix[i][j] for j in range(self.__cols) if j != column]
            for i in range(self.__rows) if i != row
        ])
        return minor

    def lu_decomposition(self) -> tuple[Matrix, Matrix]:
        n, m = self.rows_cols_count()
        l, u = Matrix(n, m, is_indentity=True), Matrix(n, m)

        for i in range(n):
            for k in range(i, n):
                total = sum(l[i][j] * u[j][k] for j in range(i))
                u[i][k] = self[i][k] - total

            for k in range(i, n):
                total = sum(l[k][j] * u[j][i] for j in range(i))
                l[k][i] = (self[k][i] - total) / u[i][i]

        return l, u

    def determinant(self) -> int | float:
        n = self.rows_cols_count()[0]

        if n == 1:
            return self[0][0]
        if n == 2:
            return self[0][0] * self[1][1] - self[0][1] * self[1][0]

        det = 0
        for i in range(n):
            for j in range(n):
                det += ((-1) ** j) * self[0][j] * self.minor(i, j).determinant()

        return det

    def inverse_matrix_lu(self) -> Matrix:
        rows, columns = self.rows_cols_count()
        indentity_matrix = Matrix(rows, columns, is_indentity=True)
        inverse_matrix = Matrix(rows, columns)

        for i in range(rows):
            indentity_column = indentity_matrix.get_column(i)
            inverse_matrix[i] = solve_lu(self, indentity_column)

        return inverse_matrix



def solve_lu(A: Matrix, b: Matrix) -> Matrix:
    if A.rows_count() != b.rows_count():
        raise ValueError('Количество строк матриц A и b должно совпадать.')

    n = b.rows_count()
    l, u = A.lu_decomposition()

    y = Matrix(n, 1)
    for i in range(n):
        total = sum(l[i][j] * y[j][0] for j in range(i))
        y[i][0] = b[i][0] - total

    x = Matrix(n, 1)
    for i in reversed(range(n)):
        total = sum(u[i][j] * x[j][0] for j in range(i + 1, n))
        x[i][0] = (y[i][0] - total) / u[i][i]

    return x


def is_equal_with_accuracy(a, b, rel_tol=1e-9, abs_tol=1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def check_solution(A: Matrix, b: Matrix, x: Matrix) -> bool:
    for row in range(A.rows_count()):
        total = 0
        for column in range(A.cols_count()):
            total += A[row][column] * x[column][0]
        if not is_equal_with_accuracy(total, b[row][0]):
            return False
    return True


def get_three_diagonals(
    matrix: Matrix
) -> tuple[list[float], list[float], list[float]]:
    a, b, c = [], [], []

    for i in range(matrix.rows_count()):
        for j in range(matrix.cols_count()):
            elem = matrix[i][j]
            if i == j + 1:
                a.append(elem)
            elif i == j:
                b.append(elem)
            elif i == j - 1:
                c.append(elem)

    return a, b, c


def thomas_algorithm(A: Matrix, d: Matrix) -> Matrix:
    n = d.rows_count()
    a, b, c = get_three_diagonals(A)

    if len(a) != n - 1 or len(b) != n or len(c) != n - 1:
        raise ValueError(
            'Неверные длины диагоналей. '
            'Ожидаемые длины: a = n - 1, b = n, c = n - 1'
        )

    alpha = [0] * n
    beta = [0] * n
    x = [0] * n

    alpha[0] = c[0] / b[0]
    beta[0] = d[0][0] / b[0]

    for i in range(1, n):
        a_i = a[i - 1] if i < n else 0
        c_i = c[i] if i < n - 1 else 0

        denominator = b[i] - a_i * alpha[i - 1]
        alpha[i] = c_i / denominator if i < n - 1 else 0
        beta[i] = (d[i][0] - a_i * beta[i - 1]) / denominator

    x[-1] = beta[-1]
    for i in reversed(range(n - 1)):
        x[i] = beta[i] - alpha[i] * x[i + 1]

    return Matrix(from_list=[[elem] for elem in x])
