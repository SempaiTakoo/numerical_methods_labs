from matrix.matrix import *
from matrix.methods import *


def print_with_breaks(text: str, argument) -> None:
    print(f'{text}:', argument, sep='\n', end='\n\n')


def calculate_task_1(path_A: str, path_b: str) -> None:
    print('=== 1.1 ===')

    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    L, U = lu_decomposition(A)
    print_with_breaks('L', L)
    print_with_breaks('U', U)

    x = solve_lu(A, b)
    print_with_breaks('solve lu x', x)
    print_with_breaks('check_solution solve lu x', check_solution(A, b, x))
    print_with_breaks('det(A)', A.determinant())
    print_with_breaks('A^(-1)', inverse_matrix_lu(A))
    print_with_breaks('A * A^(-1)', A * inverse_matrix_lu(A))


def calculate_task_2(path_A: str, path_d: str) -> None:
    print('=== 1.2 ===')

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
    print('=== 1.3 ===')

    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    x = jacobi(A, b)
    print_with_breaks('jacobi x', x)
    print_with_breaks('check solution jacobi x', check_solution(A, b, x))

    x = seidel(A, b)
    print_with_breaks('seidel x', x)
    print_with_breaks('check solution seidel x', check_solution(A, b, x))


def calculate_task_4(
    path_A: str, eps: float = 1e-9, max_iterations: int = 1000
) -> None:
    print('=== 1.4 ===')

    A = Matrix.from_file(filepath=path_A)

    print(A)

    eigenvalues, eigenvectors, errors, iterations = jacobi_rotation_method(
        A, eps=eps, max_iterations=max_iterations
    )

    print_with_breaks('Matrix A', A)
    print_with_breaks('Eigenvalues (СЗ)', eigenvalues)
    print_with_breaks('Eigenvectors (СВ) (columns)', eigenvectors)
    # print_with_breaks('Error at each iteration', errors)
    print_with_breaks('Total iterations', iterations)


def calculate_task_5(path_A: str, eps: float = 1e-9) -> None:
    print('=== 1.5 ===')

    A = Matrix(from_filepath=path_A)

    eigenvalues = qr_algorithm(A, eps)
    print_with_breaks("Matrix A", A)
    print_with_breaks("Eigenvalues (QR-algorithm)", eigenvalues)


if __name__ == '__main__':
    # python3 -m lab1.main
    calculate_task_1('lab1/examples/1_A.txt', 'lab1/examples/1_b.txt')
    calculate_task_2('lab1/examples/2_A.txt', 'lab1/examples/2_d.txt')
    calculate_task_3('lab1/examples/3_A.txt', 'lab1/examples/3_b.txt')
    calculate_task_4('lab1/examples/4_A.txt', eps=1e-9, max_iterations=100_000)
    calculate_task_5('lab1/examples/5_A.txt', eps=1e-9)
