import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)

    # Get MSE value on the validation set
    y_valid_pred = clf.predict(x_valid)
    mse = np.mean((y_valid_pred - y_valid) ** 2)
    print('Validation MSE (tau={}): {:.6f}'.format(tau, mse))

    # Plot validation predictions on top of training set
    y_train_pred = clf.predict(x_train)
    plot(x_train, y_train, y_train_pred, 'Training Set', 'output/p05b_train.png')
    plot(x_valid, y_valid, y_valid_pred, 'Validation Set', 'output/p05b_valid.png')

    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


def plot(x, y_label, y_pred, title, save_path=None):
    plt.figure()
    plt.plot(x[:, -1], y_label, 'bx', label='label')
    plt.plot(x[:, -1], y_pred, 'ro', label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left')
    if save_path is not None:
        plt.savefig(save_path)


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # LWR is non-parametric: just keep the training set; one theta is
        # computed per query point in predict().
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (l, n).

        Returns:
            Outputs of shape (l,).
        """
        # *** START CODE HERE ***
        l, n = x.shape

        # Weights w^(i) = exp(-||x_q - x^(i)||^2 / (2 tau^2)).
        # Broadcasting: (m, n) - (l, 1, n) -> (l, m, n); norm over the last
        # axis gives the (l, m) weight matrix.
        w_vector = np.exp(-np.linalg.norm(self.x - np.reshape(x, (l, 1, n)),
                                          ord=2, axis=2) ** 2 / (2 * self.tau ** 2))

        # Turn the weights into diagonal matrices, each corresponding to a
        # single query point. Shape (l, m, m).
        w = np.apply_along_axis(np.diag, axis=1, arr=w_vector)

        # Closed-form solution theta = (X^T W X)^-1 X^T W y for each query.
        # Shape (l, n), then einsum contracts it with each query x.
        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y

        return np.einsum('ij,ij->i', x, theta)
        # *** END CODE HERE ***
