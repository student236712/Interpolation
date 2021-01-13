import sympy
from sympy import *
import numpy as np
import matplotlib.pyplot as plt


class Interpolation:

    def interpolate(self, point_array=[], n=None, dic={}):
        if n is None:
            s = ""
            n = s.join(map(str, (list(range(0, len(point_array))))))
        x, x_a, x_b, P_1, P_2 = symbols('x x_a x_b P_1 P_2')
        if len(n) == 1 and n not in dic:
            for i in range(0, len(point_array)):
                dic[str(i)] = point_array[i][1]
        if n in dic:
            return dic[n]

        expr = (((x - x_a) * P_1) + ((x_b - x) * P_2)) / (x_b - x_a)
        p1 = self.interpolate(n=n[0:-1], point_array=point_array, dic=dic)
        p2 = self.interpolate(n=n[1:], point_array=point_array, dic=dic)
        xa = point_array[int(n[-1])][0]
        xb = point_array[int(n[0])][0]
        y1 = expr.subs([(x_a, xa), (x_b, xb), (P_1, p1), (P_2, p2)])
        y = simplify(y1)
        dic[n] = y
        return dic[n]


if __name__ == '__main__':
    interpolation = Interpolation()
    T = [[6, 12], [15, 5], [4, 20], [8, 6], [1, 4]]
    test_points = np.linspace(0, 16, 50)
    T_2 = np.array(T)
    y = interpolation.interpolate(point_array=T)
    latex_string = printing.latex(y)
    x = symbols('x')
    y_values = [y.subs(x, a) for a in test_points]
    x_start_data = T_2[:, 0].tolist()
    y_start_data = T_2[:, 1].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(test_points, y_values, label="Recursive")
    ax1.scatter(x_start_data, y_start_data, marker='x', c="r", label='Given points')

    ax1.legend()

    ax1.set_xlim([test_points[0], test_points[-1]])
    ax.set_xlabel("X")
    ax.set_ylabel("Polynomial value at X point")
    title = f"${latex_string}$"
    ax.set_title(title)
    plt.savefig("polynomial_comparison.png")
    plt.show()
