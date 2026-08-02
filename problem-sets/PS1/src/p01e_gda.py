import numpy as np
import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # instantiate and fit data
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot the training data and decision boundary next to the prediction file.
    plot_path = '{}.png'.format(pred_path.rsplit('.', 1)[0])
    util.plot(x_train, y_train, clf.theta, plot_path)

    # Save evaluation-set predictions.
    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    predictions = clf.predict(x_eval)
    np.savetxt(pred_path, predictions, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape

        # div
        x0 = x[y == 0]
        x1 = x[y == 1]

        # comp p(y)
        phi = len(x1) / m

        # comp miu_0, miu_1 (mean vec)
        mu_0 = np.mean(x0, axis = 0)
        mu_1 = np.mean(x1, axis = 0)

        # comp sigma, sigma_inv (cov mat)
        sigma = ((x0 - mu_0).T @ (x0 - mu_0) + (x1 - mu_1).T @ (x1 - mu_1)) / m
        sigma_inv = np.linalg.inv(sigma)

        # comp theta, theta_0
        theta = -sigma_inv @ (mu_0 - mu_1)
        theta_0 = -np.log((1 - phi) / phi) + (mu_0.T @ sigma_inv @ mu_0 - mu_1.T @ sigma_inv @ mu_1) / 2

        # merge theta_0 into theta
        self.theta = np.insert(theta, 0, theta_0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # take out theta, theta_0
        theta = self.theta[1:]
        theta_0 = self.theta[0]

        # comp natural index and get res
        # note the dim
        ind = x @ theta + theta_0
        return (ind >= 0).astype(int)
        # *** END CODE HERE

if __name__ == '__main__':
    main(
        train_path='../data/ds1_train.csv',
        eval_path='../data/ds1_valid.csv',
        pred_path='prediction/p01e_gda_pred_1.csv',
    )
    main(
        train_path='../data/ds2_train.csv',
        eval_path='../data/ds2_valid.csv',
        pred_path='prediction/p01e_gda_pred_2.csv',
    )
