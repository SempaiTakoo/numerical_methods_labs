from matrix.matrix import *


def print_with_breaks(text: str, argument) -> None:
    print(f'{text}:', argument, sep='\n', end='\n\n')


def calculate_task_1(path_A: str, path_b: str) -> None:
    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    print_with_breaks('A', A)
    print_with_breaks('b', b)

    L, U = A.lu_decomposition()
    print_with_breaks('L', L)
    print_with_breaks('U', U)

    x = solve_lu(A, b)
    print_with_breaks('U', U)
    print_with_breaks('x', x)
    print_with_breaks('check_solution A * x = b', check_solution(A, b, x))
    print_with_breaks('det(A)', A.determinant())
    print_with_breaks('A^(-1)', A.inverse_matrix_lu())
    print_with_breaks('A * A^(-1)', A * A.inverse_matrix_lu())


def calculate_task_2(path_A: str, path_d: str) -> None:
    A = Matrix(from_filepath=path_A)
    d = Matrix(from_filepath=path_d)

    a, b, c = get_three_diagonals(A)
    print_with_breaks('diagonals', (a, b, c))

    x = thomas_algorithm(A, d)
    print_with_breaks('thomas_algorithm', x)
    print_with_breaks('check solution A * x = d', check_solution(A, d, x))


def calculate_task_3() -> None:
    ...


if __name__ == '__main__':
    calculate_task_1('1_A.txt', '1_b.txt')
    calculate_task_2('2_A.txt', '2_d.txt')
