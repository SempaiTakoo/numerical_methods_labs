from matrix.matrix import *


def calculate_task_1(path_A: str, path_b: str) -> None:
    A = Matrix(from_filepath=path_A)
    b = Matrix(from_filepath=path_b)

    L, U = A.lu_decomposition()
    print(f'L:', L, sep='\n', end='\n\n')
    print(f'U:', U, sep='\n', end='\n\n')

    x = solve_lu(A, b)
    print('x:', x, sep='\n', end='\n\n')
    print('A * x = b:', check_solution(A, b, x), sep='\n', end='\n\n')
    print('det(A):', A.determinant(), sep='\n', end='\n\n')
    print('A^(-1):', A.inverse_matrix_lu(), sep='\n', end='\n\n')
    print('A * A^(-1):', A * A.inverse_matrix_lu(), sep='\n', end='\n\n')


def calculate_task_2(path_A: str, path_d: str) -> None:
    A = Matrix(from_filepath=path_A)
    d = Matrix(from_filepath=path_d)

    a, b, c = get_three_diagonals(A)
    print('diagonals:', *(a, b, c), sep='\n', end='\n\n')

    x = thomas_algorithm(A, d)
    print('thomas_algorithm:', x, sep='\n', end='\n\n')
    print('check_solution:', check_solution(A, d, x), sep='\n', end='\n\n')


if __name__ == '__main__':
    calculate_task_1('1_A.txt', '1_b.txt')
    calculate_task_2('2_A.txt', '2_d.txt')
