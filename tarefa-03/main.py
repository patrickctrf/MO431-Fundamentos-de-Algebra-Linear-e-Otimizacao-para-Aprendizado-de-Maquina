import time

import pybobyqa
from scipy.optimize import minimize, line_search
from numpy import array


def himmelblau(x):
    # Garantindo que trabalhamos com array numpy, e nao uma lista
    x = array(x)

    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau(x, *args):
    # Para x_0
    dx_0 = 2 * (2 * x[0]*(x[0] ** 2 + x[1] - 11) + x[0] + x[1] ** 2 - 7)

    # Para x_1
    dx_1 = 2 * (x[0] ** 2 + 2 * x[1] * (x[0] + x[1] ** 2 - 7) + x[1] - 11)

    return array([dx_0, dx_1])


def main():

    # =====GRADIENTE-CONJUGADO==================================================
    x = array([4, 4])
    t0 = time.time()
    val = minimize(himmelblau, x, method="CG", jac=None)
    tf = time.time()
    print("\nGRADIENTE-CONJUGADO")
    print("\niterações: ", val.nit)
    print("chamadas do gradiente: ", val.nfev)
    print("x: ", val.x)
    print("himmelblau(x): ", val.fun)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    # ======BFGS-SEM-GRADIENTE-PASSADO==========================================
    x = array([4, 4])
    t0 = time.time()
    val = minimize(himmelblau, x, method="L-BFGS-B", jac=None)
    tf = time.time()
    print("\nL-BFGS-B-sem-grad")
    print("\niterações: ", val.nit)
    print("chamadas do gradiente: ", val.nfev)
    print("x: ", val.x)
    print("himmelblau(x): ", val.fun)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    # ======BFGS-COM-GRADIENTE-PASSADO==========================================
    x = array([4, 4])
    t0 = time.time()
    val = minimize(himmelblau, x, method="L-BFGS-B", jac=grad_himmelblau)
    tf = time.time()
    print("\nL-BFGS-B-com-grad")
    print("\niterações: ", val.nit)
    print("chamadas do gradiente: ", val.nfev)
    print("x: ", val.x)
    print("himmelblau(x): ", val.fun)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    # ==========NELDER-MEAN=====================================================
    x = array([4, 4])
    t0 = time.time()
    val = minimize(himmelblau, x, method="Nelder-Mead", options={'initial_simplex':array([[-4, -4], [-4, 1], [4, -1]])})
    tf = time.time()
    print("\nNelder-mead")
    print("\niterações: ", val.nit)
    print("avaliações dos vértices do triângulo: ", val.nfev)
    print("x: ", val.x)
    print("himmelblau(x): ", val.fun)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    # =======LINE-SEARCH========================================================
    x = array([4, 4])
    x_new = x.copy()
    pk = array([-1, -1])  # -grad_himmelblau(x_new)

    t0 = time.time()
    while (1):
        alpha, fc, gc, new_fval, old_fval, new_slope = line_search(himmelblau, grad_himmelblau, x_new, pk=pk)

        # Se ainda nao convergiu, continue iterando
        if alpha is not None:
            x_new = x_new + alpha * pk
        else:
            # Quando convergir, sai do loop
            break
    tf = time.time()
    print("\nLine-Search começando na direção [-1,-1]")
    print("\niterações: ", fc)
    print("chamadas do gradiente: ", gc)
    print("x: ", x_new)
    print("himmelblau(x): ", old_fval)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    x = array([4, 4])
    x_new = x.copy()
    pk = -grad_himmelblau(x_new)

    t0 = time.time()
    while (1):
        alpha, fc, gc, new_fval, old_fval, new_slope = line_search(himmelblau, grad_himmelblau, x_new, pk=pk)

        # Se ainda nao convergiu, continue iterando
        if alpha is not None:
            x_new = x_new + alpha * pk
        else:
            # Quando convergir, sai do loop
            break
    tf = time.time()
    print("\nLine-Search começando na direção oposta ao gradiente")
    print("\niterações: ", fc)
    print("chamadas do gradiente: ", gc)
    print("x: ", x_new)
    print("himmelblau(x): ", old_fval)
    print("Tempo demandado pela otimização [s]: ", tf - t0)
    print("\n")

    # ===========BOBYQA=========================================================
    x = array([4, 4])
    print("\nBOBYQA")
    # Estabelecendo os limites (lower <= x <= upper)
    lower = array([-10.0, -10.0])
    upper = array([10.0, 10.0])
    # Executa a minimizacao
    t0 = time.time()
    val = pybobyqa.solve(himmelblau, x, bounds=(lower, upper))
    tf = time.time()

    # Imprime resultados
    print(val)
    print("Tempo demandado pela otimização [s]: ", tf - t0)



if __name__ == "__main__":
    main()
