import matplotlib.pyplot as plt
import numpy.random
import pandas


class Exercise:
    def __init__(self, name):
        self.name = name
        numpy.random.seed(42)
        self.sample_size = 1000
        self.multivariate_sample = numpy.random.normal(0.0, 1.0, self.sample_size)

    def generate_sample(self):
        # Multivariate Distribution
        mean = [0, 0]
        cov = [[1, -0.5], [-0.5, 2]]
        self.multivariate_sample = numpy.random.multivariate_normal(mean, cov, self.sample_size)
        x = self.multivariate_sample[:, 0]
        y = self.multivariate_sample[:, 1]
        plt.scatter(x, y)

    def calculate_sample_mean(self):
        return self.get_sample_mean(self.multivariate_sample)

    def calculate_sample_covariance_matrix(self):
        return self.get_sample_covariance_matrix(self.multivariate_sample)

    @staticmethod
    def get_sample_mean(data):
        n = len(data)
        p = len(data[0])
        return [sum([element[variate] for element in data]) / n for variate in range(p)]

    @staticmethod
    def get_sample_covariance_matrix(data):
        n = len(data)
        p = len(data[0])

        sample_mean = Exercise.get_sample_mean(data)

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
    analyzer = Exercise('PyCharm')
    analyzer.generate_sample()
    plt.show()

    sample_mean = analyzer.calculate_sample_mean()
    sample_covariance = analyzer.calculate_sample_covariance_matrix()

    print(sample_mean)
    print(sample_covariance)
