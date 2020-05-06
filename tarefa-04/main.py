import numpy as np
from pandas import DataFrame
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

    # ==============GRID-SEARCH===============================================

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

    # ==============fim-GRID-SEARCH===========================================
