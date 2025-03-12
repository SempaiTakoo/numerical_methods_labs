from matrix.matrix import *


def print_with_breaks(text: str, argument) -> None:
    print(f'{text}:', argument, sep='\n', end='\n\n')


def calculate_task_1(path_A: str, path_b: str) -> None:
    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    L, U = A.lu_decomposition()
    print_with_breaks('L', L)
    print_with_breaks('U', U)

    x = solve_lu(A, b)
    print_with_breaks('solve lu x', x)
    print_with_breaks('check_solution solve lu x', check_solution(A, b, x))
    print_with_breaks('det(A)', A.determinant())
    print_with_breaks('A^(-1)', A.inverse_matrix_lu())
    print_with_breaks('A * A^(-1)', A * A.inverse_matrix_lu())


def calculate_task_2(path_A: str, path_d: str) -> None:
    A = Matrix(from_filepath=path_A)
    d = Matrix(from_filepath=path_d)

    a, b, c = get_three_diagonals(A)
    print_with_breaks('diagonals', (a, b, c))

    print_with_breaks('A', A)
    print_with_breaks('b', d)

    x = thomas_algorithm(A, d)
    print_with_breaks('thomas x', x)
    print_with_breaks('check solution thomas x', check_solution(A, d, x))


def calculate_task_3(path_A: str, path_b: str) -> None:
    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    x = jacobi(A, b)
    print_with_breaks('jacobi x', x)
    print_with_breaks('check solution jacobi x', check_solution(A, b, x))

    x = seidel(A, b)
    print_with_breaks('seidel x', x)
    print_with_breaks('check solution seidel x', check_solution(A, b, x))


if __name__ == '__main__':
    calculate_task_1('1_A.txt', '1_b.txt')
    calculate_task_2('2_A.txt', '2_d.txt')
    calculate_task_3('3_A.txt', '3_b.txt')
