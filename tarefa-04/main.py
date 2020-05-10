import cma
import hyperopt
import numpy as np
from hyperopt import fmin, tpe, hp

from pyswarm import pso
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR


# Retorna o MAE pra o SVR
def evaluate_svr(args):
    gamma, C, epsilon = args

    gamma = 2 ** gamma
    C = 2 ** C

    svr_object = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
    svr_object.fit(x_treino, y_treino)
    return mean_absolute_error(svr_object.predict(x_teste), y_teste)


# Retorna o MAE tambem, mas recebendo argumentos como array
def erro(x):
    svr_object = SVR(kernel='rbf', gamma=2 ** x[0], C=2 ** x[1], epsilon=x[2])
    svr_object.fit(x_treino, y_treino)
    return mean_absolute_error(svr_object.predict(x_teste), y_teste)


# Como hp nao possui distribuicoes uniformes nos expoentes de base dois, fazemos
# este processo na funcao de avaliacao
def evaluate_svr_hp(args):
    gamma, C, epsilon = args

    gamma = 2 ** gamma
    C = 2 ** C

    svr_object = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
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

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
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

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
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

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    # Definindo o espaco em que iremos trabalhar. Gamma e C serao elevados a 2
    # na funcao de avaliacao do SVR
    space = [hp.uniform('gamma', -15, 3),
             hp.uniform('C', -5, 15),
             hp.uniform('epsilon', 0.05, 1.0)]

    # calling the hyperopt function
    resultado = fmin(fn=evaluate_svr_hp, space=space, algo=tpe.suggest, max_evals=125)

    print("\n---------------------OTIMIZAÇÃO-BAYESIANA-hyperopt---------------------")

    print("\nMelhor conjunto de parâmetros: \n")
    print("C: ", 2 ** resultado["C"])

    print("gamma: ", 2 ** resultado["gamma"])

    print("epsilon: ", resultado["epsilon"])

    # Calculando o modelo encontrado para verificar seu erro mean_absolute_error
    svr_object = SVR(kernel='rbf', gamma=2 ** resultado["gamma"], C=2 ** resultado["C"], epsilon=resultado["epsilon"])
    svr_object.fit(x_treino, y_treino)
    mae = mean_absolute_error(svr_object.predict(x_teste), y_teste)

    print("\nMAE teste: \n", mae)

    # ==============fim-OTIMIZACAO-BAYESIANA====================================

    # ======================PSO=================================================

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    lb = [-15, -5, 0.05]
    ub = [3, 15, 1.0]

    xopt, fopt = pso(erro, lb, ub, maxiter=11, swarmsize=11)

    print("\n---------------------PSO---------------------")

    # Calculando o modelo encontrado para verificar seu erro mean_absolute_error
    svr_object = SVR(kernel='rbf', gamma=2 ** xopt[0], C=2 ** xopt[1], epsilon=xopt[2])
    svr_object.fit(x_treino, y_treino)
    mae = mean_absolute_error(svr_object.predict(x_teste), y_teste)

    print("\nMAE teste: \n", mae)
    print("C: ", 2 ** xopt[1])
    print("epsilon: ", xopt[2])
    print("gamma: ", 2 ** xopt[0])

    # ======================fim-PSO=============================================

    # ======================SIMULATED-ANNEALING=================================

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    # Valores limite para a busca em cada um dos parametros
    lw = [-15, -5, 0.05]
    up = [3, 15, 1.0]

    resultado = dual_annealing(evaluate_svr, bounds=list(zip(lw, up)), no_local_search=True, maxiter=125)

    print("\n---------------------SIMULATED-ANNEALING---------------------")

    # Calculando o modelo encontrado para verificar seu erro mean_absolute_error
    svr_object = SVR(kernel='rbf', gamma=2 ** resultado.x[0], C=2 ** resultado.x[1], epsilon=resultado.x[2])
    svr_object.fit(x_treino, y_treino)
    mae = mean_absolute_error(svr_object.predict(x_teste), y_teste)

    print("\nMAE teste: \n", mae)

    print("C: ", 2 ** resultado.x[1])

    print("gamma: ", 2 ** resultado.x[0])

    print("epsilon: ", resultado.x[2])

    # ======================fim-SIMULATED-ANNEALING=============================

    # ======================CMA-ES==============================================

    # Fixando a semente para garantir resultados aleatorios reprodutiveis.
    np.random.seed(1234)

    opts = cma.CMAOptions()
    opts.set("bounds", [[-15, -5, 0.05], [3, 15, 1.0]])
    opts.set('maxfevals', 125)
    parametros, es = cma.fmin2(evaluate_svr, [1, 1, 1], 0.1, opts)

    # Calculando o modelo encontrado para verificar seu erro mean_absolute_error
    svr_object = SVR(kernel='rbf', gamma=2 ** parametros[0], C=2 ** parametros[1], epsilon=parametros[2])
    svr_object.fit(x_treino, y_treino)
    mae = mean_absolute_error(svr_object.predict(x_teste), y_teste)

    print("\nMAE teste: \n", mae)

    print("C: ", 2 ** parametros[1])

    print("gamma: ", 2 ** parametros[0])

    print("epsilon: ", parametros[2])

    # ======================fim-CMA-ES==========================================
