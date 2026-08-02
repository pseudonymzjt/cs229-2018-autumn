import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Plot training-set labels vs. predictions.
    # util.plot is for classification decision boundaries, so use a
    # label-vs-prediction scatter instead.
    plot_path = '{}.png'.format(pred_path.rsplit('.', 1)[0])
    y_train_pred = clf.predict(x_train)
    plt.figure()
    plt.plot(y_train, 'go', label='label')
    plt.plot(y_train_pred, 'rx', label='prediction')
    plt.suptitle('Training Set', fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig(plot_path)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Batch gradient ascent on the full log-likelihood.
        # Gradient: (1/m) * X^T (y - exp(X theta)); update theta += lr * gradient.
        # (Part (c)'s stochastic rule is the same per-example gradient without
        # the 1/m averaging; the dataset's y-values are huge, so averaging is
        # what keeps lr = 1e-7 stable here.)
        m, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n, dtype=float)

        step = self.step_size / m * x.T @ (y - np.exp(x @ self.theta))
        while np.linalg.norm(step, 1) >= self.eps:
            self.theta = self.theta + step
            step = self.step_size / m * x.T @ (y - np.exp(x @ self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
