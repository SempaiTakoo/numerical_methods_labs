from __future__ import annotations
import math

from .utils import is_equal_with_accuracy


FLOAT_ROUND = 7


class Matrix:

    def __init__(
        self,
        rows: int = 0,
        cols: int = 0,
        is_indentity: bool = False,
        from_list: None | list[list[float]] = None,
        from_filepath: str = None
    ) -> None:
        if from_list:
            self.__matrix = from_list.copy()
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

    @staticmethod
    def from_list(lst: list[list[float]]) -> Matrix:
        assert len(lst) != 0 and len(lst[0]) != 0
        return Matrix(rows=len(lst), cols=len(lst[0]), from_list=lst)

    @staticmethod
    def from_file(filepath: str) -> Matrix | None:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                rows, cols = map(int, file.readline().split(' '))
                list_matrix = [
                    list(map(float, file.readline().split(' ')))
                    for _ in range(rows)
                ]
                if rows != len(list_matrix[0]) or cols != len(list_matrix[0]):
                    raise ValueError
                return Matrix.from_list(list_matrix)

        except FileNotFoundError:
            print(f'Файл {filepath} не найден.')
        except ValueError:
            print(f'Файл {filepath} содержит некорректные данные.')
        except Exception as e:
            print(f'Произошла ошибка: {e}')

    @staticmethod
    def indentity(n: int) -> Matrix:
        return Matrix.from_list(
            [[int(i == j) for j in range(n)] for i in range(n)]
        )

    def copy(self) -> Matrix:
        lst = [row[:] for row in self.__matrix]
        return Matrix.from_list(lst)

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
            print('Файл содержит некорректные данные.')
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
                str(round(self.__matrix[i][j], FLOAT_ROUND)).rjust(max_num_width)
                for j in range(self.__cols)
            )
            for i in range(self.__rows)
        )

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

    def __add__(self, other: Matrix) -> Matrix:
        if (
            self.rows_count() != other.rows_count()
            or self.cols_count() != other.cols_count()
        ):
            raise ValueError('При сложении размеры матриц должны совпадать.')
        return Matrix(from_list=[
            [self[i][j] + other[i][j] for j in range(self.cols_count())]
            for i in range(self.rows_count())
        ])

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

    def get_column(self, index: int) -> Matrix:
        return Matrix(from_list=[
            [self[row][index]] for row in range(self.__rows)
        ])

    def minor(self, row: int, column: int) -> Matrix:
        minor = Matrix(from_list=[
            [self.__matrix[i][j] for j in range(self.__cols) if j != column]
            for i in range(self.__rows) if i != row
        ])
        return minor

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

    def is_symmetric(self) -> bool:
        if self.rows_count() != self.cols_count():
            return False
        for i in range(self.__rows):
            for j in range(i + 1, self.__cols):
                if not is_equal_with_accuracy(self[i][j], self[j][i]):
                    return False
        return True

    def transpose(self) -> Matrix:
        transposed = [
            [self[j][i] for j in range(self.__rows)]
            for i in range(self.__cols)
        ]
        return Matrix(from_list=transposed)
