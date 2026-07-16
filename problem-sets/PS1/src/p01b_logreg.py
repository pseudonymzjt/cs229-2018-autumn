import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'problem-sets/PS1/src/prediction/p01b_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    eps = 1 * 10 ** -5
    def fit(self, x: np.array, y: np.array):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            h = 1 / (1 + np.exp(-x @ self.theta))
            grad = x.T @ (h - y) / m
            H = x.T @ np.diag(h * (1 - h)) @ x / m
            self.theta -= np.linalg.inv(H) @ grad

            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            if self.verbose:
                print(f"Iter {i + 1:3d} | Loss: {loss:.6f} | ||grad||: {np.linalg.norm(grad, ord=1):.6e}")

            if np.linalg.norm(grad, ord=1) < self.eps:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations.")
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta)) >= 0.5
        # *** END CODE HERE ***

main("E:\Sophomore\cs229-2018-autumn\problem-sets\PS1\data\ds1_train.csv", "E:\Sophomore\cs229-2018-autumn\problem-sets\PS1\data\ds1_valid.csv", "E:\Sophomore\cs229-2018-autumn\problem-sets\PS1\src\prediction\p01b.csv")