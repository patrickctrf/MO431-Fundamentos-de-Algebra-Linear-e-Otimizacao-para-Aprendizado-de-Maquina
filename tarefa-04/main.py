import hyperopt
import numpy as np
from hyperopt import fmin, tpe, hp

from pandas import DataFrame
from pyswarm import pso
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR

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

    # return the mean squared error
    def evaluate_svr(args):
        gamma, C, epsilon = args
        svr_object = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
        svr_object.fit(x_treino, y_treino)
        return mean_absolute_error(svr_object.predict(x_teste), y_teste)


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

    # Funcao de avaliacao para o erro da funcao
    def erro(x):
        svr_object = SVR(kernel='rbf', gamma=x[0], C=x[1], epsilon=x[2])
        svr_object.fit(x_treino, y_treino)
        return mean_absolute_error(svr_object.predict(x_teste), y_teste)


    lb = [2 ** (-15), 2 ** (-5), 0.05]
    ub = [2 ** 3, 2 ** 15, 1.0]

    xopt, fopt = pso(erro, lb, ub, maxiter=11, swarmsize=11)

    print("\nMAE: ", fopt)
    print("C: ", xopt[1])
    print("epsilon: ", xopt[2])
    print("gamma: ", xopt[0])

    # ======================fim-PSO=============================================
