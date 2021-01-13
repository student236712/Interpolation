from sympy import *
import numpy as np
import matplotlib.pyplot as plt


class Interpolation:

    def interpolate_recursively(self, point_array=[], indexing_string=None, memo_object={}):
        if indexing_string is None:
            s = ""
            indexing_string = s.join(map(str, (list(range(0, len(point_array))))))
        if len(indexing_string) == 1 and indexing_string not in memo_object:
            for i in range(0, len(point_array)):
                memo_object[str(i)] = point_array[i][1]
        if indexing_string in memo_object:
            return memo_object[indexing_string]

        x, x_a, x_b, P_1, P_2 = symbols('x x_a x_b P_1 P_2')

        # Given expression for calculating polynomial recursively
        expr = (((x - x_a) * P_1) + ((x_b - x) * P_2)) / (x_b - x_a)

        # Get the value of 1'st parent polynomial - indexes 0 to pre-last
        p1 = self.interpolate_recursively(point_array=point_array, indexing_string=indexing_string[0:-1],
                                          memo_object=memo_object)
        # Get the value of 2'nd parent polynomial - indexes 1 to last
        p2 = self.interpolate_recursively(point_array=point_array, indexing_string=indexing_string[1:],
                                          memo_object=memo_object)
        xa = point_array[int(indexing_string[-1])][0]
        xb = point_array[int(indexing_string[0])][0]
        y1 = expr.subs([(x_a, xa), (x_b, xb), (P_1, p1), (P_2, p2)])
        y = simplify(y1)
        memo_object[indexing_string] = y
        return memo_object[indexing_string]


if __name__ == '__main__':
    interpolation = Interpolation()
    points_to_interpolate = [[6, 12], [15, 5], [4, 20], [8, 6], [1, 4]]
    test_points = np.linspace(0, 16, 50)
    T_2 = np.array(points_to_interpolate)
    x_start_data = T_2[:, 0].tolist()
    y_start_data = T_2[:, 1].tolist()

    interpolated_function = interpolation.interpolate_recursively(point_array=points_to_interpolate)
    latex_string = printing.latex(interpolated_function)

    x = symbols('x')
    y_values = [interpolated_function.subs(x, a) for a in test_points]

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
