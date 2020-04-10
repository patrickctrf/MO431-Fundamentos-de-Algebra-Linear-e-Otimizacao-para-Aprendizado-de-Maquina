from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def rosenbrock_3d(x):
    """
Funcao de Rosenbrock em 3d
    :param x: Array de ordenadas da funcao (x1, x2, x3).
    :return: Retorna o valor da funcao no ponto (x[0], x[1], x[2]).
    """
    x = np.array(x)

    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 100 * (x[2] - x[1] ** 2) ** 2 + (1 - x[1]) ** 2


def tolerancia(x, x_old):
    """
Computa a tolerancia para o gradiente. Serve para indicar quando o gradiente
esta proximo do minimo e ja esta parando de deslocar o ponto indicado.
    :param x: ponto atual gerado pelo gradiente
    :param x_old: ponto anterior do caminho sendo percorrido pelo gradiente
    :return: Razao entre a variacao do ponto atual para o anterior.
    """
    x = np.array(x)
    x_old = np.array(x_old)

    return norm(x - x_old) / norm(x_old)


def gradiente_rosenbrock_3d(x, step_derivativo=10 ** -10):
    """
Computa o gradiente da funcao rosenbrock 3d
    :param x: Ponto a avaliar o gradiente.
    :param step_derivativo: Largura do intervalo a calcular as derivadas parciais
    :return: Gradiente avaliado no ponto x (x1, x2, x3).
    """
    x = np.array(x)

    df1 = (rosenbrock_3d(x + np.array([step_derivativo, 0, 0])) - rosenbrock_3d(x)) / step_derivativo
    df2 = (rosenbrock_3d(x + np.array([0, step_derivativo, 0])) - rosenbrock_3d(x)) / step_derivativo
    df3 = (rosenbrock_3d(x + np.array([0, 0, step_derivativo])) - rosenbrock_3d(x)) / step_derivativo

    return np.array([df1, df2, df3])


def minimizacao_gradiente_explicito(lr=10 ** -3, max_passos=20000):
    """
Realiza a minimizacao por gradiente convencional (SGD sem momento) e plota os
resultados.
    :param lr: Learning Rate (deafult: 10 ** -3).
    :param max_passos: Limite de iteracoes para minimizacao (default: 20000).
    """

    # Ponto inicial da fncao onde calcularemos o gradiente
    x = np.array([0, 0, 0])
    # Para calcular a variacao de um ponto "x" em relacao ao anterior, devemos
    # salvar o anteriror.
    x_old = np.copy(x)
    # Uma lista para guardar os valores de cada iteracao e poder plotar os
    # resultado ao final.
    plot_rosenbrock = list()
    # Uma lista para guardar os valores de "x" em cada iteracao e poder plotar
    # os resultado ao final.
    plot_x = list()

    # A funcao de tolerancia daria erro na primeira iteracao por estarmos
    # sobre o ponto zero. Fazemos portanto a primeira iteracao fora do loop e
    # sem calcular a tolerancia.
    plot_rosenbrock.append(rosenbrock_3d(x))
    plot_x.append(np.copy(x))
    x = x - lr * gradiente_rosenbrock_3d(x)

    # Neste loop repetimos a sequencia de salvar o ponto anterior, calcular o
    # gradiente, usa-lo para calcular o novo ponto e verificar o criterio de
    # tolerancia. Temos um limite de iteracoes em "max_passos", caso o gradiente
    # comece a divergir, mas a intencao eh que o limite de tolerancia seja
    # batido e a instrucao "break" rompa o loop.
    for i in range(max_passos):
        # Salva ponto anterior
        x_old = np.copy(x)

        # Salva os dados para posterior plotagem
        plot_rosenbrock.append(rosenbrock_3d(x))
        plot_x.append(np.copy(x))
        # Usa o gradiente para plotar os novos pontos.
        x = x - lr * gradiente_rosenbrock_3d(x)

        # Verifica o criterio de tolerancia
        if tolerancia(x, x_old) < 10 ** -4:
            # Se ja estamos abaixo da tolerancia, salva o ultimo dado e encerra.
            plot_rosenbrock.append(rosenbrock_3d(x))
            break

    # Plotamos os valores de f(x) para cada atualizacao de "x".
    plt.title("Valores para cálculo do gradiente explícito com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("f(x) - Rosenbrock 3d")
    plt.plot(plot_rosenbrock)
    plt.show()

    # Exibimos na tela as informacoes de interesse.
    print("\nNúmero de passos do gradiente: ", len(plot_rosenbrock))
    print("\nValor da função no mínimo local: ", rosenbrock_3d(x))
    print("Valores de x1: ", x[0], ", x2: ", x[1], "e x3: ", x[2])

    # Tranformamos em uma matriz numpy para poder usar a API de indices.
    plot_x = np.array(plot_x)

    # Abaixo plotamos os valores das ordenadas "x" para entendermos como as
    # "hipoteses" do gradiente se comportam.
    plt.title("Valores para cálculo do gradiente explícito com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x1")
    plt.plot(plot_x[:, 0].reshape(plot_x[:, 0].shape[0]))
    plt.show()

    plt.title("Valores para cálculo do gradiente explícito com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x2")
    plt.plot(plot_x[:, 1].reshape(plot_x[:, 0].shape[0]))
    plt.show()

    plt.title("Valores para cálculo do gradiente explícito com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x3")
    plt.plot(plot_x[:, 2].reshape(plot_x[:, 0].shape[0]))
    plt.show()


def minimizacao_gradiente_tensorflow(lr=10 ** -3, max_passos=20000):
    """
Realiza a minimizacao por gradiente convencional (SGD sem momento) usando a API
do tensorflow.
    :param lr: Learning Rate (deafult: 10 ** -3).
    :param max_passos: Maximo de iteracoes a realizar para a otimizacao (default: 20000).
    """

    # Para calcular a tolerancia, temos que saber quanto os pontos de ordenadas
    # "x" valiam antes. Para isto que servem estas variaveis.
    x1_old, x2_old, x3_old = tf.Variable(0), tf.Variable(0), tf.Variable(0)

    # Precisam ser variaveis do tensorflow para o graidiente poder alterar o
    # valor da funcao e computar as derivadas.
    x1, x2, x3 = tf.Variable(10 ** -10), tf.Variable(10 ** -10), tf.Variable(10 ** -10)

    # Selecionamos o tipo de otimizador, que sera o gradiente comum, uma vez que
    # o SGD sem momento iguala a um gradiente simples.
    opt = tf.keras.optimizers.SGD(lr=lr)

    # Salvamos aqui os valores de "x", doas gradientes e da propria funcao f(x)
    # para pode plotar e analisar apos a otimizacao.
    lista_plot = list()

    # Se a iteracao chegar ao numero "max_passos", significa que o calculo
    # divergiu e devemos parar. A intencao eh que a tolerancia de minimizacao
    # seja atingida antes e a instrucao break seja chamada.
    for i in range(max_passos):
        # A funcao escolhida para avaliar as derivadas eh o GradientTape
        with tf.GradientTape() as tape:
            # A funcao sobre a qual calculamos os gradientes (rosenbrock 3d).
            y = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 + 100 * (x3 - x2 ** 2) ** 2 + (1 - x2) ** 2
            # Obtemos os valores dos gradientes no ponto "x".
            grads = tape.gradient(y, [x1, x2, x3])
            # Colocamos eles em uma nova lista.
            processed_grads = [g for g in grads]
            # Zipamos os gradientes e os valores de "x" juntos para computar os
            # proximos valores de "x".
            grads_and_vars = zip(processed_grads, [x1, x2, x3])

        # Coloamos na lista de registros de dados para o grafico os valores
        # desta iteracao.
        lista_plot.append([y.numpy(),
                           x1.numpy(),
                           x2.numpy(),
                           x3.numpy(),
                           grads[0].numpy(),
                           grads[1].numpy(),
                           grads[2].numpy()]
                          )

        # Atualizamos as variaveis que guardam a iteracao anterior para poder
        # gerar o novo ponto "x".
        x1_old, x2_old, x3_old = x1.numpy(), x2.numpy(), x3.numpy()
        # Obtemos os novos valores de "x" atraves do otimizador escolhido (SGD).
        opt.apply_gradients(grads_and_vars)

        # Se a diferenca relativaentre o novo ponto gerado e o antigo for menor
        # que o especificado, estamos proximos o suficiente do minimo e paramos
        # aqui.
        if tolerancia([x1, x2, x3], [x1_old, x2_old, x3_old]) < 10 ** -4:
            break

    # Tranformamos a lista de registros em numpy array para poder usar sua API.
    lista_plot = np.array(lista_plot)

    # Exibimos o grafico de saida com os valores da funcao a cada atualizacao de
    # "x".
    plt.title("Valores para cálculo do gradiente tensorflow com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("f(x) - Rosenbrock 3d")
    plt.plot(lista_plot[:, 0].reshape(lista_plot[:, 0].shape[0]))
    plt.show()

    # Informamos os valores de interesse. 1 passo somado a mais devido ao
    # deslocamento do ponto zero que fazemos incialmente para resolver a
    # singularidade da regiao.
    print("\nNúmero de passos do gradiente: ", lista_plot.shape[0] + 1)
    print("\nValor da função no mínimo local: ", rosenbrock_3d([x1.numpy(), x2.numpy(), x3.numpy()]))
    print("\nValores de x1: ", x1.numpy(), ", x2: ", x2.numpy(), "e x3: ", x3.numpy())


    # Abaixo plotamos os valores das ordenadas "x" para entendermos como as
    # "hipoteses" do gradiente se comportam.
    plt.title("Valores para cálculo do gradiente tensorflow com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x1")
    plt.plot(lista_plot[:, 1].reshape(lista_plot[:, 0].shape[0]))
    plt.show()

    plt.title("Valores para cálculo do gradiente tensorflow com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x2")
    plt.plot(lista_plot[:, 2].reshape(lista_plot[:, 0].shape[0]))
    plt.show()

    plt.title("Valores para cálculo do gradiente tensorflow com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("Valor de x3")
    plt.plot(lista_plot[:, 3].reshape(lista_plot[:, 0].shape[0]))
    plt.show()


def main():
    # Learning rate pequeno aumenta a chance de convergir, ao passo que diminui
    # a velocidade em que se alcanca o minimo local.
    minimizacao_gradiente_explicito(lr=1.00 * 10 ** -4)

    # Learning rate um pouco mais alto permite diminuir o tempo de convergencia.
    minimizacao_gradiente_explicito(lr=1.00 * 10 ** -3)

    # Learning rate alto demais causando a extrapolacao do minimo local e a
    # divergencia do gradiente.
    minimizacao_gradiente_explicito(lr=1.62 * 10 ** -3)

    minimizacao_gradiente_tensorflow(lr=1.00 * 10 ** -4)

    minimizacao_gradiente_tensorflow(lr=1.00 * 10 ** -3)


if __name__ == "__main__":
    main()
