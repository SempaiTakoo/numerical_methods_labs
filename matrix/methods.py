import math

from .matrix import Matrix
from .utils import is_equal_with_accuracy


def lu_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix]:
    n, m = matrix.rows_cols_count()
    l, u = Matrix(n, m, is_indentity=True), Matrix(n, m)

    for i in range(n):
        for k in range(i, n):
            total = sum(l[i][j] * u[j][k] for j in range(i))
            u[i][k] = matrix[i][k] - total

        for k in range(i, n):
            total = sum(l[k][j] * u[j][i] for j in range(i))
            # Реализовать выбор главного элемента
            # Помнить число перестановок
            l[k][i] = (matrix[k][i] - total) / u[i][i]

    return l, u


def solve_lu(A: Matrix, b: Matrix) -> Matrix:
    if A.rows_count() != b.rows_count():
        raise ValueError("Количество строк матриц A и b должно совпадать.")

    n = b.rows_count()
    l, u = lu_decomposition(A)

    y = Matrix(n, 1)
    for i in range(n):
        total = sum(l[i][j] * y[j][0] for j in range(i))
        y[i][0] = b[i][0] - total

    x = Matrix(n, 1)
    for i in reversed(range(n)):
        total = sum(u[i][j] * x[j][0] for j in range(i + 1, n))
        x[i][0] = (y[i][0] - total) / u[i][i]

    return x


def inverse_matrix_lu(matrix: Matrix) -> Matrix:
    rows, columns = matrix.rows_cols_count()
    indentity_matrix = Matrix(rows, columns, is_indentity=True)
    inverse_matrix = Matrix(rows, columns)

    for i in range(rows):
        indentity_column = indentity_matrix.get_column(i)
        inverse_matrix[i] = solve_lu(matrix, indentity_column)

    return inverse_matrix


def check_solution(A: Matrix, b: Matrix, x: Matrix) -> bool:
    for row in range(A.rows_count()):
        total = 0
        for column in range(A.cols_count()):
            total += A[row][column] * x[column][0]
        if not is_equal_with_accuracy(total, b[row][0]):
            return False
    return True


def get_three_diagonals(matrix: Matrix) -> tuple[list[float], list[float], list[float]]:
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
            "Неверные длины диагоналей. Ожидаемые длины: a = n - 1, b = n, c = n - 1"
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


def jacobi(
    A: Matrix, b: Matrix, tol: float = 1e-9, max_iterations: int = 1000
) -> Matrix:
    n = b.rows_count()
    x = Matrix(n, 1)

    for k in range(max_iterations):
        x_new = Matrix(n, 1)
        for i in range(n):
            sigma = sum(A[i][j] * x[j][0] for j in range(n) if i != j)
            x_new[i][0] = (b[i][0] - sigma) / A[i][i]

        diff = max(abs(x_new[i][0] - x[i][0]) for i in range(n))
        if diff < tol:
            break

        x = x_new.copy()

    return x


def seidel(
    A: Matrix, b: Matrix, tol: float = 1e-9, max_iterations: int = 1000
) -> Matrix:
    n = b.rows_count()
    x = Matrix(n, 1)

    for _ in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j][0] for j in range(i))
            sum2 = sum(A[i][j] * x[j][0] for j in range(i + 1, n))
            x_new[i][0] = (b[i][0] - sum1 - sum2) / A[i][i]

        converge = all(abs(x_new[i][0] - x[i][0]) < tol for i in range(n))
        if converge:
            break

        x = x_new

    return x


def compute_rotation_params(
    a_pp: float, a_qq: float, a_pq: float
) -> tuple[float, float]:
    """
    Вычисляет косинус и синус угла поворота для метода Якоби.
    """
    if a_pq == 0:
        return 1.0, 0.0

    tau = (a_qq - a_pp) / (2 * a_pq)
    t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1 + tau**2))
    cos_phi = 1.0 / math.sqrt(1 + t**2)
    sin_phi = t * cos_phi
    return cos_phi, sin_phi


def off_diagonal_norm(A: Matrix) -> float:
    """
    Считает норму внедиагональных элементов: sqrt(Σ A[i][j]^2 для i ≠ j)
    """
    n = A.rows_count()
    return math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j))


def jacobi_rotation_method(
    A: Matrix, eps: float = 1e-9, max_iterations: int = 1000
) -> tuple[list[float], Matrix, list[float], int]:
    """
    Метод вращений Якоби для симметричных матриц.
    Возвращает:
        список собственных значений,
        матрицу собственных векторов (по столбцам),
        список норм ошибок,
        число итераций.
    """
    n = A.rows_count()
    if not A.is_symmetric():
        raise ValueError("Матрица должна быть симметричной.")

    A = A.copy()
    eigenvectors = Matrix(n, n, is_indentity=True)

    iterations = 0
    errors = []

    def max_offdiag(a: Matrix) -> tuple[float, int, int]:
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > abs(max_val):
                    max_val = a[i][j]
                    p, q = i, j
        return max_val, p, q

    while iterations < max_iterations:
        norm = off_diagonal_norm(A)
        errors.append(norm)
        if norm < eps:
            break

        _, p, q = max_offdiag(A)
        a_pp, a_qq, a_pq = A[p][p], A[q][q], A[p][q]
        cos_phi, sin_phi = compute_rotation_params(a_pp, a_qq, a_pq)

        for i in range(n):
            if i != p and i != q:
                aip = A[i][p]
                aiq = A[i][q]
                A[i][p] = A[p][i] = cos_phi * aip - sin_phi * aiq
                A[i][q] = A[q][i] = sin_phi * aip + cos_phi * aiq

        A[p][p] = cos_phi**2 * a_pp - 2 * sin_phi * cos_phi * a_pq + sin_phi**2 * a_qq
        A[q][q] = sin_phi**2 * a_pp + 2 * sin_phi * cos_phi * a_pq + cos_phi**2 * a_qq
        A[p][q] = A[q][p] = 0.0

        for i in range(n):
            vip = eigenvectors[i][p]
            viq = eigenvectors[i][q]
            eigenvectors[i][p] = cos_phi * vip - sin_phi * viq
            eigenvectors[i][q] = sin_phi * vip + cos_phi * viq

        iterations += 1

    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, eigenvectors, errors, iterations


def qr_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """
    Классическое QR-разложение с использованием модифицированного метода Грамма-Шмидта.
    Возвращает Q (ортонормированная) и R (верхнетреугольная) матрицы.
    """
    n, m = A.rows_cols_count()
    Q_columns = []
    R = Matrix(m, m)

    for j in range(m):
        v = A.get_column(j)

        for i in range(j):
            q_i = Q_columns[i]
            r_ij = sum(v[k][0] * q_i[k][0] for k in range(n))
            R[i][j] = r_ij
            # v = v - r_ij * q_i
            for k in range(n):
                v[k][0] -= r_ij * q_i[k][0]

        norm = math.sqrt(sum(v[k][0] ** 2 for k in range(n)))
        if norm == 0:
            raise ValueError(
                "Невозможно построить ортонормальный базис (линейно зависимые векторы)."
            )

        R[j][j] = norm
        q_j = Matrix(from_list=[[v[k][0] / norm] for k in range(n)])
        Q_columns.append(q_j)

    Q = Matrix(n, m)
    for j in range(m):
        for i in range(n):
            Q[i][j] = Q_columns[j][i][0]

    return Q, R


def qr_algorithm(
    A: Matrix, eps: float = 1e-9, max_iterations: int = 1000
) -> list[float]:
    """
    Итеративный QR-алгоритм для приближенного нахождения собственных значений.
    Работает для произвольной квадратной матрицы.
    """
    if A.rows_count() != A.cols_count():
        raise ValueError("Матрица должна быть квадратной.")

    n = A.rows_count()
    A_k = A.copy()

    for iteration in range(max_iterations):
        Q, R = qr_decomposition(A_k)
        A_next = R * Q

        max_delta = max(abs(A_next[i][i] - A_k[i][i]) for i in range(n))

        A_k = A_next

        if max_delta < eps:
            break

    eigenvalues = [A_k[i][i] for i in range(n)]
    return eigenvalues


def simple_iteration_method(
    phi: callable, x0: float, eps: float = 1e-9, max_iterations: int = 1000
) -> tuple[float, list[float], int]:
    """
    Метод простой итерации: x_{n+1} = φ(x_n)
    """
    x = x0
    errors = []

    for i in range(max_iterations):
        x_next = phi(x)
        error = abs(x_next - x)
        errors.append(error)

        if error < eps:
            return x_next, errors, i + 1
        x = x_next

    raise RuntimeError("Метод простой итерации не сошелся.")


def newton_method(
    f: callable, df: callable, x0: float, eps: float = 1e-9, max_iterations: int = 1000
) -> tuple[float, list[float], int]:
    """
    Метод Ньютона: x_{n+1} = x_n - f(x_n)/f'(x_n)
    """
    x = x0
    errors = []

    for i in range(max_iterations):
        f_x = f(x)
        df_x = df(x)

        if abs(df_x) < 1e-12:
            raise ZeroDivisionError("Производная близка к нулю.")

        x_next = x - f_x / df_x
        error = abs(x_next - x)
        errors.append(error)

        if error < eps:
            return x_next, errors, i + 1
        x = x_next

    raise RuntimeError("Метод Ньютона не сошелся.")


def simple_iteration_system(
    phi: callable,
    x0: Matrix,
    eps: float = 1e-9,
    max_iterations: int = 1000
) -> tuple[Matrix, list[float], int]:
    """
    Метод простой итерации для систем: x_{k+1} = φ(x_k)
    """
    x = x0.copy()
    errors = []

    for k in range(max_iterations):
        x_next = phi(x)
        diff = max(abs(x_next[i][0] - x[i][0]) for i in range(x.rows_count()))
        errors.append(diff)

        if diff < eps:
            return x_next, errors, k + 1

        x = x_next.copy()

    raise RuntimeError("Метод простой итерации не сошелся.")


def newton_system(
    F: callable,
    J: callable,
    x0: Matrix,
    eps: float = 1e-9,
    max_iterations: int = 1000
) -> tuple[Matrix, list[float], int]:
    """
    Метод Ньютона для систем F(x) = 0.
    """
    x = x0.copy()
    errors = []

    for k in range(max_iterations):
        Fx = F(x)
        Jx = J(x)

        dx = solve_lu(Jx, Fx * -1)
        x_next = x + dx

        diff = max(abs(dx[i][0]) for i in range(dx.rows_count()))
        errors.append(diff)

        if diff < eps:
            return x_next, errors, k + 1

        x = x_next.copy()

    raise RuntimeError("Метод Ньютона не сошелся.")


def lagrange_interpolation(X: list[float], Y: list[float], x_star: float) -> float:
    """
    Интерполяция Лагранжа: P(x*) = sum(Y_i * L_i(x*))
    """
    n = len(X)
    result = 0.0

    for i in range(n):
        term = Y[i]
        for j in range(n):
            if i != j:
                term *= (x_star - X[j]) / (X[i] - X[j])
        result += term

    return result


def newton_interpolation(X: list[float], Y: list[float], x_star: float) -> float:
    """
    Интерполяция Ньютона с разделёнными разностями.
    """
    n = len(X)
    coef = Y.copy()

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (X[i] - X[i - j])

    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_star - X[i]) + coef[i]

    return result


def cubic_spline_interpolation(X: list[float], Y: list[float], x_star: float) -> float:
    """
    Построение естественного кубического сплайна и вычисление значения в точке x_star.
    """
    n = len(X) - 1
    h = [X[i + 1] - X[i] for i in range(n)]

    A = Matrix(n - 1, n - 1)
    d = Matrix(n - 1, 1)

    for i in range(1, n):
        A[i - 1][i - 1] = (h[i - 1] + h[i]) / 3
        if i - 2 >= 0:
            A[i - 1][i - 2] = h[i - 1] / 6
        if i < n - 1:
            A[i - 1][i] = h[i] / 6

        d[i - 1][0] = (Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1]

    c_internal = thomas_algorithm(A, d)
    c = [0.0] + [c_internal[i][0] for i in range(n - 1)] + [0.0]

    for i in range(n):
        if X[i] <= x_star <= X[i + 1]:
            break
    else:
        raise ValueError("x_star вне интервала интерполяции.")

    xi, xi1 = X[i], X[i + 1]
    hi = h[i]
    yi, yi1 = Y[i], Y[i + 1]
    ci, ci1 = c[i], c[i + 1]

    dx = x_star - xi

    term1 = ci / (6 * hi) * (xi1 - x_star) ** 3
    term2 = ci1 / (6 * hi) * (x_star - xi) ** 3
    term3 = (yi / hi - ci * hi / 6) * (xi1 - x_star)
    term4 = (yi1 / hi - ci1 * hi / 6) * (x_star - xi)

    return term1 + term2 + term3 + term4


def least_squares_polynomial(X: list[float], Y: list[float], degree: int) -> Matrix:
    """
    Решает нормальную систему для МНК полинома заданной степени.
    Возвращает коэффициенты [a_0, a_1, ..., a_degree]^T
    """
    n = degree + 1
    m = len(X)

    A = Matrix(n, n)
    b = Matrix(n, 1)

    for i in range(n):
        for j in range(n):
            A[i][j] = sum(X[k] ** (i + j) for k in range(m))
        b[i][0] = sum(Y[k] * (X[k] ** i) for k in range(m))

    return solve_lu(A, b)


def evaluate_polynomial(coeffs: Matrix, x: float) -> float:
    """
    Вычисляет значение многочлена в точке x по заданным коэффициентам.
    """
    return sum(coeffs[i][0] * (x ** i) for i in range(coeffs.rows_count()))
