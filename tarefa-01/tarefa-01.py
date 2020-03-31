import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD

# ================Lendo-arquivo-NPY=============================================

X = np.load("X.npy")

# ================Imprimindo-imagem-primeira pessoa=============================
plt.imshow(X[0].reshape(50, 37), cmap=cm.gray)
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

# ======Verifique-a-formulação-Padrao-do-SVD====================================

# Calculamos a diferenca entre cada elemento da matriz original e da matriz
# reconstruida a partir do SVD compacto.
matriz_de_diferencas = X - np.matmul(np.matmul(U, D), Vh)

print("\nMáximo Erro SVD Padrão: " + str(np.absolute(matriz_de_diferencas).max()))
print("Erro Médio SVD Padrão: " + str(np.absolute(matriz_de_diferencas).mean()))
print("Razão entre Máximo Erro e Erro Médio SVD Padrão: " + str(np.absolute(matriz_de_diferencas).max() / np.absolute(matriz_de_diferencas).mean()))

# ===================Matriz-Reduzida-SVD-Padrao=================================

k = 100
matriz_reduzida_full_matrices = np.matmul(U[:, :k], D[:k, :k])

print("\nFormato da matriz reduzida SVD full matrices: " + str(matriz_reduzida_full_matrices.shape))

# ===================Matriz-Reconstruida-SVD-Padrao=============================

matriz_reconstruida_full_matrices = np.matmul(matriz_reduzida_full_matrices, Vh[:k, :])

print("\nFormato da matriz reconstruida SVD full matrices: " + str(matriz_reconstruida_full_matrices.shape))

# ===================Exibindo-na-Tela-Matriz-Reconstruida-SVD-full-matrices=====

plt.imshow(matriz_reconstruida_full_matrices[0].reshape(50, 37), cmap=cm.gray)
plt.show()

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

# ======Verifique-a-formulação-compacta-do-SVD==================================

# Calculamos a diferenca entre cada elemento da matriz original e da matriz
# reconstruida a partir do SVD compacto.
matriz_de_diferencas = X - np.matmul(np.matmul(U, D), Vh)

print("\nMáximo Erro SVD Compacto: " + str(np.absolute(matriz_de_diferencas).max()))
print("Erro Médio SVD Compacto: " + str(np.absolute(matriz_de_diferencas).mean()))
print("Razão entre Máximo Erro e Erro Médio SVD Compacto: " + str(np.absolute(matriz_de_diferencas).max() / np.absolute(matriz_de_diferencas).mean()))

# ===================Matriz-Reduzida-SVD-Compacto===============================

k = 100
matriz_reduzida_svd_compacto = np.matmul(U[:, :k], D[:k, :k])

print("\nFormato da matriz reduzida a partir do SVD compacto: " + str(matriz_reduzida_svd_compacto.shape))

# ===================Matriz-Reconstruida-SVD-Compacto===========================

matriz_reconstruida_svd_compacto = np.matmul(matriz_reduzida_svd_compacto, Vh[:k, :])

print("\nFormato da matriz reconstruida a partir do SVD compacto: " + str(matriz_reconstruida_svd_compacto.shape))

# ====================Full-matrices-VS-Compacto-SVD=============================

# Calculamos a diferenca entre cada elemento da matriz original e da matriz
# reconstruida a partir de cada SVD
matriz_de_diferencas_svd_full_matrices = X - matriz_reconstruida_full_matrices
matriz_de_diferencas_svd_compacto = X - matriz_reconstruida_svd_compacto

print("\nMáximo Erro SVD Compacto: " + str(np.absolute(matriz_de_diferencas_svd_full_matrices).max()))
print("Erro Médio SVD Compacto: " + str(np.absolute(matriz_de_diferencas_svd_full_matrices).mean()))

print("\nMáximo Erro SVD Compacto: " + str(np.absolute(matriz_de_diferencas_svd_compacto).max()))
print("Erro Médio SVD Compacto: " + str(np.absolute(matriz_de_diferencas_svd_compacto).mean()))

# ===================Exibindo-na-Tela-Matriz-Reconstruida=======================

plt.imshow(matriz_reconstruida_svd_compacto[0].reshape(50, 37), cmap=cm.gray)
plt.show()

# ===================Truncated-SVD-SKLEARN======================================

# Inicia/Gera o modelo do SVD, mas so faz calculo dps do "fit"
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)

# Podemos obter a matriz reduzida (U*D).
matriz_reduzida_UD = svd.fit_transform(X)
# Tambem desnecessario, so pra obter a matriz V^-1
Vh = svd.components_

# Gera REALMENTE as matrizes do SVD (nem tinhamos passado a matriz original ate
# aqui).
svd.fit(X)

# Recupera a matrix original apos ter convertido para o SVD.
matriz_reconstruida_sklearn = svd.inverse_transform(matriz_reduzida_UD)

print("\nFormatos das matrizes TruncatedSVD")
print("Matriz reduzida (UD): ", str(matriz_reduzida_UD.shape))
print("Matriz V^-1: ", str(Vh.shape))

# ======Verifique-a-formulação-compacta-do-TruncatedSVD-sklearn=================

# Calculamos a diferenca entre cada elemento da matriz original e da matriz
# reconstruida a partir do TruncatedSVD do sklearn.
matriz_de_diferencas = X - matriz_reconstruida_sklearn

print("\nMáximo Erro TruncatedSVD do sklearn: " + str(np.absolute(matriz_de_diferencas).max()))
print("Erro Médio TruncatedSVD do sklearn: " + str(np.absolute(matriz_de_diferencas).mean()))
print("Razão entre Máximo Erro e Erro Médio TruncatedSVD do sklearn: " + str(np.absolute(matriz_de_diferencas).max() / np.absolute(matriz_de_diferencas).mean()))

# ===================Exibindo-na-Tela-Matriz-Reconstruida-TruncatedSVD-sklear===

plt.imshow(matriz_reconstruida_sklearn[0].reshape(50, 37), cmap=cm.gray)
plt.show()
