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

    def interpolate_iteratively(self, point_array=[], n=0):
        interpolation_points = np.array(point_array)
        x_data_series = interpolation_points[:, 0].tolist()
        y_data_series = interpolation_points[:, 1].tolist()

        final_val = 0
        for j in range(0, len(y_data_series)):
            numerator_of_the_fraction = 1
            denominator_of_the_fraction = 1
            for i in range(0, len(x_data_series)):
                if i != j:
                    numerator_of_the_fraction *= n - x_data_series[i]
                    denominator_of_the_fraction *= x_data_series[j] - x_data_series[i]
            final_val += numerator_of_the_fraction / denominator_of_the_fraction * y_data_series[j]
        return final_val


if __name__ == '__main__':
    interpolation = Interpolation()
    test_points = np.linspace(0, 20, 50)
    plt.plot()
    plt.ylim([-10, 10])
    plt.xlim([0, 20])
    points_to_interpolate = plt.ginput(5)
    T_2 = np.array(points_to_interpolate)
    x_start_data = T_2[:, 0].tolist()
    y_start_data = T_2[:, 1].tolist()

    interpolated_function = interpolation.interpolate_recursively(point_array=points_to_interpolate)
    latex_string = printing.latex(interpolated_function)

    x = symbols('x')
    y_recursive_values = [interpolated_function.subs(x, a) for a in test_points]
    y_iterative_values = [interpolation.interpolate_iteratively(point_array=points_to_interpolate, n=a) for a in
                          test_points]
    print(points_to_interpolate)

    plt.plot(test_points, y_recursive_values, '--', label="Recursive")
    plt.plot(test_points, y_iterative_values, '.', label="Iterative")
    plt.scatter(x_start_data, y_start_data, marker='x', c="k", label='Given points')
    plt.title(f"${latex_string}$")
    plt.xlabel('x')
    plt.ylabel('Polynomial value at X point')
    plt.legend()
    plt.savefig("polynomial_comparison_2.png")
    plt.show()
