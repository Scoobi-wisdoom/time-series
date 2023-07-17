# This is a sample Python script.
import matplotlib.pyplot as plt
import numpy.random
import pandas


def exercise(name):
    numpy.random.seed(42)
    sample_size = 1000
    multivariate_sample = numpy.random.normal(0.0, 1.0, sample_size)

    # sample value calculation
    mean = numpy.mean(multivariate_sample)

    series = pandas.Series(multivariate_sample)

    # time series plot
    # plt.plot(series.index, series.values)

    # histogram plot
    # plt.hist(series.values, bins=30, alpha=0.7, edgecolor='black')

    # scatter plot of noises with trend line added: y(i) = a * x(i) + b + e(i)
    # a = 1.1
    # b = 1.0
    # x_values = numpy.linspace(0, 10, len(series))
    # y_values = a * x_values + b + series.values
    # plt.scatter(x_values, y_values)
    #
    # # Calculate the LMSE line
    # slope, intercept = numpy.polyfit(x_values, y_values, 1)
    # lmse_line = slope * x_values + intercept
    # plt.plot(x_values, lmse_line, color='red', label='LMSE Line')

    # # cumulative sum
    # cumulative_sum = numpy.cumsum(series)
    # plt.plot(series.index, cumulative_sum)

    # Multivariate Distribution
    mean = [0, 0]
    cov = [[1, -0.5], [-0.5, 2]]
    multivariate_sample = numpy.random.multivariate_normal(mean, cov, sample_size)
    x = multivariate_sample[:, 0]
    y = multivariate_sample[:, 1]
    plt.scatter(x, y)
    # plt.show()

    sample_mean = get_sample_mean(multivariate_sample)
    sample_covariance = get_sample_covariance_matrix(multivariate_sample)
    print(sample_mean)  # [0.026526494624065602, 0.05323189171822471]
    print(numpy.mean(multivariate_sample, axis=0))  # [0.02652649 0.05323189]
    print()
    print(sample_covariance)  # [[ 0.96164838 -0.41312129] [-0.41312129  1.92668612]]
    print(numpy.cov(multivariate_sample, rowvar=False,
                    bias=False))  # [[ 0.96164838 -0.41312129]  [-0.41312129  1.92668612]]


def get_sample_mean(data):
    n = len(data)
    p = len(data[0])
    return [sum([element[variate] for element in data]) / n for variate in range(p)]


def get_sample_covariance_matrix(data):
    n = len(data)
    p = len(data[0])

    sample_mean = get_sample_mean(data)

    sample_covariance_matrix = numpy.zeros((p, p))
    for i in range(n):
        parameter_vector = []
        for j in range(p):
            parameter_vector.append(data[i][j])

        residual_vector = numpy.array(parameter_vector) - sample_mean
        sample_covariance_matrix += numpy.outer(residual_vector, residual_vector)

    return sample_covariance_matrix / (n - 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    exercise('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
