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


def sgd_manual(lr=10 ** -3, max_passos=20000):
    x = np.array([0, 0, 0])
    x_old = np.copy(x)
    plot_rosenbrock = list()

    plot_rosenbrock.append(rosenbrock_3d(x))
    x = x - lr * gradiente_rosenbrock_3d(x)

    for i in range(max_passos):
        x_old = np.copy(x)

        plot_rosenbrock.append(rosenbrock_3d(x))
        x = x - lr * gradiente_rosenbrock_3d(x)

        if tolerancia(x, x_old) < 10 ** -4:
            plot_rosenbrock.append(rosenbrock_3d(x))
            break
    plt.title("Gradiente manual com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("f(x) - Rosenbrock 3d")
    plt.plot(plot_rosenbrock)
    plt.show()

    print("\nNúmero de passos do gradiente: ", len(plot_rosenbrock))
    print("\nValor da função no mínimo local: ", rosenbrock_3d(x))


def sgd_tensorflow(lr=10 ** -3, max_passos=20000):
    x1_old, x2_old, x3_old = tf.Variable(0), tf.Variable(0), tf.Variable(0)
    x1, x2, x3 = tf.Variable(10 ** -10), tf.Variable(10 ** -10), tf.Variable(10 ** -10)

    opt = tf.keras.optimizers.SGD(lr=lr)

    lista_plot = list()

    for i in range(max_passos):
        with tf.GradientTape() as tape:
            y = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 + 100 * (x3 - x2 ** 2) ** 2 + (1 - x2) ** 2
        grads = tape.gradient(y, [x1, x2, x3])
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, [x1, x2, x3])

        lista_plot.append([y.numpy(), x1.numpy(), x2.numpy(), x3.numpy(), grads[0].numpy(), grads[1].numpy(), grads[2].numpy()])

        x1_old, x2_old, x3_old = x1.numpy(), x2.numpy(), x3.numpy()
        opt.apply_gradients(grads_and_vars)

        if tolerancia([x1, x2, x3], [x1_old, x2_old, x3_old]) < 10 ** -4:
            break

    lista_plot = np.array(lista_plot)

    plt.title("Gradiente tensorflow com LR de " + str(lr))
    plt.xlabel("Número de atualizações de x")
    plt.ylabel("f(x) - Rosenbrock 3d")
    plt.plot(lista_plot[:, 0].reshape(lista_plot[:, 0].shape[0]))
    plt.show()

    print("\nNúmero de passos do gradiente: ", lista_plot.shape[0])
    print("\nValor da função no mínimo local: ", rosenbrock_3d([x1, x2, x3]).numpy())

    return lista_plot


def main():
    # # Learning rate pequeno aumenta a chance de convergir, ao passo que diminui
    # # a velocidade em que se alcanca o minimo local.
    # sgd_manual(lr=1.00 * 10 ** -4)
    #
    # # Learning rate um pouco mais alto permite diminuir o tempo de convergencia.
    # sgd_manual(lr=1.00 * 10 ** -3)
    #
    # # Learning rate alto demais causando a extrapolacao do minimo local e a
    # # divergencia do gradiente.
    # sgd_manual(lr=1.62 * 10 ** -3)

    sgd_tensorflow(lr=1.00 * 10 ** -4)

    sgd_tensorflow(lr=1.00 * 10 ** -3)


if __name__ == "__main__":
    main()
