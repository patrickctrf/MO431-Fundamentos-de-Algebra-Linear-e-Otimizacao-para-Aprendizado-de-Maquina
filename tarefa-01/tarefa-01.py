import numpy as np
import matplotlib.pyplot as plt

# ================Lendo-arquivo-NPY=============================================

X = np.load("X.npy")

# ================Imprimindo-imagem-primeira pessoa=============================
plt.imshow(X[0].reshape(50, 37))
plt.show()

# ================Fatoração-SVD-Padrao==========================================
U, D, Vh = np.linalg.svd(X, full_matrices=True)

# Para comportar a matriz recuperada "D", precisa ser multiplicavel pelo numero
# de colunas de U e numero de linhas de Vh.
matrix_auxiliar = np.zeros((U.shape[1], Vh.shape[0]))

# Substituimos a diagonal principal pelos devido elementos do vetor D para
# retomar a matriz D esparsa.
for i in range(min(matrix_auxiliar.shape)):
    matrix_auxiliar[i][i] = D[i]

# Retomamaos a matriz D.
D = matrix_auxiliar

print("\nFormatos das matrizes SVD padrão")
print("Matriz U: ", str(U.shape))
print("Matriz D: ", str(D.shape))
print("Matriz V^-1: ", str(Vh.shape))

# ================Fatoração-SVD-Compacto========================================
U, D, Vh = np.linalg.svd(X, full_matrices=False)

# Para comportar a matriz recuperada "D", precisa ser multiplicavel pelo numero
# de colunas de U e numero de linhas de Vh.
matrix_auxiliar = np.zeros((U.shape[1], Vh.shape[0]))

# Substituimos a diagonal principal pelos devido elementos do vetor D para
# retomar a matriz D esparsa.
for i in range(min(matrix_auxiliar.shape)):
    matrix_auxiliar[i][i] = D[i]

# Retomamaos a matriz D.
D = matrix_auxiliar

print("\nFormatos das matrizes SVD Compacto")
print("Matriz U: ", str(U.shape))
print("Matriz D: ", str(D.shape))
print("Matriz V^-1: ", str(Vh.shape))

# ======Verifiquea-formulação-compacta-do-SVD===================================

# Calculamos a diferenca entre cada elemento da matriz original e da matriz
# reconstruida a partir do SVD compacto.
matriz_de_diferencas = X - np.matmul(np.matmul(U, D), Vh)

print("\nMáximo Erro: " + str(matriz_de_diferencas.max()))
print("Erro Médio: " + str(matriz_de_diferencas.mean()))
print("Razão entre Máximo Erro e Erro Médio: " + str(matriz_de_diferencas.max() / matriz_de_diferencas.mean()))

x = 1

# ===================Matriz-Reduzida============================================

k = 100
matriz_reduzida = np.matmul(U[:, :k], D[:k, :k])

print("\nFormato da matriz reduzida: " + str(matriz_reduzida.shape))

# ===================Matriz-Reconstruida========================================

matriz_reconstruida = np.matmul(matriz_reduzida, Vh[:k, :])

print("\nFormato da matriz reconstruida: " + str(matriz_reconstruida.shape))

# ===================Exibindo-na-Tela-Matriz-Reconstruida=======================

plt.imshow(matriz_reconstruida[0].reshape(50, 37))
plt.show()
