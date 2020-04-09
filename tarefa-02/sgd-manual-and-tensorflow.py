from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt


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

    plt.xlabel("Número de atualizações de x")
    plt.ylabel("f(x) - Rosenbrock 3d")
    plt.plot(plot_rosenbrock)
    plt.show()

    print("\nNúmero de passos do gradiente: ", len(plot_rosenbrock))


def main():
    # Learning rate pequeno aumenta a chance de convergir, ao passo que diminui
    # a velocidade em que se alcanca o minimo local.
    sgd_manual(lr=1.00 * 10 ** -4)

    # Learning rate um pouco mais alto permite diminuir o tempo de convergencia.
    sgd_manual(lr=1.00 * 10 ** -3)

    # Learning rate alto demais causando a extrapolacao do minimo local e a
    # divergencia do gradiente.
    sgd_manual(lr=1.62 * 10 ** -3)


if __name__ == "__main__":
    main()
