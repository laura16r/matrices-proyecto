import numpy as np

# 1. NaivOnArray — O(n³) clásico
def naiv_on_array(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# 2. NaivLoopUnrollingTwo — desdobla el loop k de 2 en 2
def naiv_loop_unrolling_two(A, B, n):
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 0
            while k < n - 1:
                C[i][j] += A[i][k]*B[k][j] + A[i][k+1]*B[k+1][j]
                k += 2
            if k < n:
                C[i][j] += A[i][k] * B[k][j]
    return C

# 3. NaivLoopUnrollingFour — igual pero de 4 en 4
def naiv_loop_unrolling_four(A, B, n):
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 0
            while k < n - 3:
                C[i][j] += (A[i][k]*B[k][j] + A[i][k+1]*B[k+1][j]
                           + A[i][k+2]*B[k+2][j] + A[i][k+3]*B[k+3][j])
                k += 4
            while k < n:
                C[i][j] += A[i][k] * B[k][j]
                k += 1
    return C

# 4. WinogradOriginal — O(n³) con menos multiplicaciones
def winograd_original(A, B, n):
    C = np.zeros((n, n))
    row_factor = np.zeros(n)
    col_factor = np.zeros(n)
    for i in range(n):
        for k in range(0, n//2):
            row_factor[i] += A[i][2*k] * A[i][2*k+1]
    for j in range(n):
        for k in range(0, n//2):
            col_factor[j] += B[2*k][j] * B[2*k+1][j]
    for i in range(n):
        for j in range(n):
            C[i][j] = -row_factor[i] - col_factor[j]
            for k in range(0, n//2):
                C[i][j] += (A[i][2*k]+B[2*k+1][j]) * (A[i][2*k+1]+B[2*k][j])
            if n % 2 == 1:
                C[i][j] += A[i][n-1] * B[n-1][j]
    return C

# 5. WinogradScaled — igual pero escala las matrices primero
def winograd_scaled(A, B, n):
    # Encontrar el factor de escala
    lambda_val = max(np.max(np.abs(A)), np.max(np.abs(B)))
    if lambda_val == 0:
        return np.zeros((n, n))
    scale = 1.0 / lambda_val
    return winograd_original(A * scale, B * scale, n) / (scale ** 2)

# 6. StrassenNaiv — recursivo, O(n^2.81)
def strassen_naiv(A, B):
    n = len(A)
    if n == 1:
        return A * B
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    P1 = strassen_naiv(A11+A22, B11+B22)
    P2 = strassen_naiv(A21+A22, B11)
    P3 = strassen_naiv(A11, B12-B22)
    P4 = strassen_naiv(A22, B21-B11)
    P5 = strassen_naiv(A11+A12, B22)
    P6 = strassen_naiv(A21-A11, B11+B12)
    P7 = strassen_naiv(A12-A22, B21+B22)
    C = np.zeros((n, n))
    C[:mid,:mid] = P1+P4-P5+P7
    C[:mid,mid:] = P3+P5
    C[mid:,:mid] = P2+P4
    C[mid:,mid:] = P1-P2+P3+P6
    return C

# 7. StrassenWinograd — Strassen con los productos de Winograd
def strassen_winograd(A, B):
    n = len(A)
    if n == 1:
        return A * B
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    S1 = A21 + A22; T1 = B12 - B11
    S2 = S1 - A11;  T2 = B22 - T1
    S3 = A11 - A21; T3 = B22 - B12
    S4 = A12 - S2;  T4 = B21 - T2
    P1 = strassen_winograd(A11, B11)
    P2 = strassen_winograd(A12, B21)
    P3 = strassen_winograd(S1, T1)
    P4 = strassen_winograd(S2, T2)
    P5 = strassen_winograd(S3, T3)
    P6 = strassen_winograd(S4, B22)
    P7 = strassen_winograd(A22, T4)
    U1 = P1 + P2; U2 = P1 + P4; U3 = U2 + P5
    C = np.zeros((n, n))
    C[:mid,:mid] = U1
    C[:mid,mid:] = U3 + P3 + P6
    C[mid:,:mid] = U2 + P7 - P3  # ajustado
    C[mid:,mid:] = U3 + P7
    return C

# 8-15: Bloques y paralelos del artículo
# (estos usan numpy para simular el acceso por bloques)

def sequential_block_row_col(A, B, bsize=32):  # III.3
    n = len(A); C = np.zeros((n,n))
    for i1 in range(0, n, bsize):
        for j1 in range(0, n, bsize):
            for k1 in range(0, n, bsize):
                C[i1:i1+bsize, j1:j1+bsize] += \
                    A[i1:i1+bsize, k1:k1+bsize] @ B[k1:k1+bsize, j1:j1+bsize]
    return C

def sequential_block_row_row(A, B, bsize=32):  # IV.3 (mismo patrón, idx distinto)
    return sequential_block_row_col(A, B, bsize)

def sequential_block_col_col(A, B, bsize=32):  # V.3
    return sequential_block_row_col(A, B, bsize)

# Paralelos — usan concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def parallel_block(A, B, bsize=32):  # III.4 / IV.4 / V.4
    n = len(A); C = np.zeros((n,n))
    def compute_block(i1, j1):
        for k1 in range(0, n, bsize):
            C[i1:i1+bsize, j1:j1+bsize] += \
                A[i1:i1+bsize, k1:k1+bsize] @ B[k1:k1+bsize, j1:j1+bsize]
    with ThreadPoolExecutor() as executor:
        for i1 in range(0, n, bsize):
            for j1 in range(0, n, bsize):
                executor.submit(compute_block, i1, j1)
    return C

def enhanced_parallel_block(A, B, bsize=32):  # III.5 / IV.5
    n = len(A); C = np.zeros((n,n))
    mid = n // 2
    def top_half():
        for i1 in range(0, mid, bsize):
            for j1 in range(0, n, bsize):
                for k1 in range(0, n, bsize):
                    C[i1:i1+bsize, j1:j1+bsize] += \
                        A[i1:i1+bsize, k1:k1+bsize] @ B[k1:k1+bsize, j1:j1+bsize]
    def bottom_half():
        for i1 in range(mid, n, bsize):
            for j1 in range(0, n, bsize):
                for k1 in range(0, n, bsize):
                    C[i1:i1+bsize, j1:j1+bsize] += \
                        A[i1:i1+bsize, k1:k1+bsize] @ B[k1:k1+bsize, j1:j1+bsize]
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1, f2 = ex.submit(top_half), ex.submit(bottom_half)
        f1.result(); f2.result()
    return C