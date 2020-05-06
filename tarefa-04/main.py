import cma
import hyperopt
import numpy as np
from hyperopt import fmin, tpe, hp

from pandas import DataFrame
from pyswarm import pso
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR


# Retorna o MAE pra o SVR
def evaluate_svr(args):
    gamma, C, epsilon = args
    svr_object = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
    svr_object.fit(x_treino, y_treino)
    return mean_absolute_error(svr_object.predict(x_teste), y_teste)


# Retorna o MAE tambem, mas recebendo argumentos como array
def erro(x):
    svr_object = SVR(kernel='rbf', gamma=x[0], C=x[1], epsilon=x[2])
    svr_object.fit(x_treino, y_treino)
    return mean_absolute_error(svr_object.predict(x_teste), y_teste)


if __name__ == '__main__':
    # Dados de treino
    x_treino = np.load("Xtreino5.npy")
    y_treino = np.load("ytreino5.npy")

    # Dados de teste
    x_teste = np.load("Xteste5.npy")
    y_teste = np.load("yteste5.npy")

    # ==============RANDOM-SEARCH===============================================

    # Fixando a emenete para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    c = 2 ** np.random.uniform(-5, 15, 125)
    gamma = 2 ** np.random.uniform(-15, 3, 125)
    epsilon = np.random.uniform(0.05, 1.0, 125)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'C': c, 'gamma': gamma, 'epsilon': epsilon}

    svr_object = SVR()
    # Montamos o objeto que realiza a busca
    randomized_search_engine = \
        RandomizedSearchCV(
            estimator=svr_object,
            param_distributions=parametros,
            scoring="neg_mean_absolute_error"
        )

    # Realizamos a busca atraves do treinamento
    randomized_search_engine.fit(x_treino, y_treino)

    # Predizemos os valores de teste para avaliar o resultado.
    mae = mean_absolute_error(randomized_search_engine.predict(x_teste), y_teste)

    print("\n---------------------RANDOM_SEARCH_CV---------------------")

    print("\nMelhor conjunto de parâmetros: \n", randomized_search_engine.best_estimator_)

    print("\nMAE teste: \n", mae)

    # ==============fim-RANDOM-SEARCH===========================================

    # ==============GRID-SEARCH=================================================

    # Fixando a emenete para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    c = 2 ** np.random.uniform(-5, 15, 5)
    gamma = 2 ** np.random.uniform(-15, 3, 5)
    epsilon = np.random.uniform(0.05, 1.0, 5)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'C': c, 'gamma': gamma, 'epsilon': epsilon}

    svr_object = SVR()
    # Montamos o objeto que realiza a busca
    randomized_search_engine = \
        GridSearchCV(
            estimator=svr_object,
            param_grid=parametros,
            scoring="neg_mean_absolute_error"
        )

    # Realizamos a busca atraves do treinamento
    randomized_search_engine.fit(x_treino, y_treino)

    # Predizemos os valores de teste para avaliar o resultado.
    mae = mean_absolute_error(randomized_search_engine.predict(x_teste), y_teste)

    print("\n---------------------GRID_SEARCH_CV---------------------")

    print("\nMelhor conjunto de parâmetros: \n", randomized_search_engine.best_estimator_)

    print("\nMAE teste: \n", mae)

    # ==============fim-GRID-SEARCH=============================================

    # ==============OTIMIZACAO-BAYESIANA========================================

    # Definindo o espaco em que iremos trabalhar
    space = [hp.uniform('gamma', 2 ** (-15), 2 ** 3),
             hp.uniform('C', 2 ** (-5), 2 ** (15)),
             hp.uniform('epsilon', 0.05, 1.0)]

    # calling the hyperopt function
    resultado = fmin(fn=evaluate_svr, space=space, algo=tpe.suggest, max_evals=125)

    print("\n---------------------OTIMIZAÇÃO-BAYESIANA-hyperopt---------------------")

    print("\nMelhor conjunto de parâmetros: \n", resultado)

    # Calculando o modelo encontrado para verificar seu erro mean_absolute_error
    svr_object = SVR(kernel='rbf', gamma=resultado["gamma"], C=resultado["C"], epsilon=resultado["epsilon"])
    svr_object.fit(x_treino, y_treino)
    mae = mean_absolute_error(svr_object.predict(x_teste), y_teste)

    print("\nMAE teste: \n", mae)

    # ==============fim-OTIMIZACAO-BAYESIANA====================================

    # ======================PSO=================================================

    lb = [2 ** (-15), 2 ** (-5), 0.05]
    ub = [2 ** 3, 2 ** 15, 1.0]

    xopt, fopt = pso(erro, lb, ub, maxiter=11, swarmsize=11)

    print("\n---------------------PSO---------------------")

    print("\nMAE: ", fopt)
    print("C: ", xopt[1])
    print("epsilon: ", xopt[2])
    print("gamma: ", xopt[0])

    # ======================fim-PSO=============================================

    # ======================SIMULATED-ANNEALING=================================

    # Valores limite para a busca em cada um dos parametros
    lw = [2 ** (-15), 2 ** (-5), 0.05]
    up = [2 ** 3, 2 ** (15), 1.0]

    # Valores iniciais para a busca
    values = [1, 1, 1]

    resultado = dual_annealing(evaluate_svr, bounds=list(zip(lw, up)), no_local_search=True, maxiter=125)

    print("\n---------------------SIMULATED-ANNEALING---------------------")

    print("\nMAE: ", resultado.fun)

    print("C: ", resultado.x[1])

    print("gamma: ", resultado.x[0])

    print("epsilon: ", resultado.x[2])

    # ======================fim-SIMULATED-ANNEALING=============================

    # ======================CMA-ES==============================================

    opts = cma.CMAOptions()
    opts.set("bounds", [[2 ** (-15), 2 ** (-5), 0.05], [2 ** 3, 2 ** (15), 1.0]])
    opts.set('maxfevals', 125)
    parametros, es = cma.fmin2(evaluate_svr, [1, 1, 1], 0.1, opts)

    print("\n---------------------CMA-ES---------------------")

    print("\nMAE: ", es.result.fbest)

    print("C: ", parametros[1])

    print("gamma: ", parametros[0])

    print("epsilon: ", parametros[2])

    # ======================fim-CMA-ES==========================================
