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


    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary next to the prediction file.
    plot_path = '{}.png'.format(pred_path.rsplit('.', 1)[0])
    util.plot(x_train, y_train, model.theta, plot_path)

    # Save predictions
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred >= 0.5, fmt='%d')


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    @staticmethod
    def _sigmoid(z):
        """Compute the sigmoid function without overflowing for large |z|."""
        z = np.asarray(z, dtype=float)
        probabilities = np.empty_like(z)
        positive = z >= 0

        probabilities[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        probabilities[~positive] = exp_z / (1.0 + exp_z)
        return probabilities

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if x.ndim != 2:
            raise ValueError('x must be a 2D array.')
        if x.shape[0] == 0:
            raise ValueError('x must contain at least one example.')
        if y.shape[0] != x.shape[0]:
            raise ValueError('x and y must contain the same number of examples.')

        m, n = x.shape
        # Problem 1(b) specifies Newton's method starts from theta = 0.
        self.theta = np.zeros(n, dtype=float)

        for iteration in range(self.max_iter):
            scores = x @ self.theta
            probabilities = self._sigmoid(scores)
            gradient = x.T @ (probabilities - y) / m

            # diag(weights) is never formed: this is O(m*n) rather than O(m^2)
            # in memory and avoids the explicit inverse of the Hessian.
            weights = probabilities * (1.0 - probabilities)
            hessian = x.T @ (weights[:, np.newaxis] * x) / m
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Degenerate features can make the Hessian singular. The
                # least-squares Newton step is the stable fallback.
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            next_theta = self.theta - step
            update_norm = np.linalg.norm(next_theta - self.theta, ord=1)
            self.theta = next_theta

            if self.verbose:
                # logaddexp(0, z) - y*z is the stable logistic-loss expression.
                loss = np.mean(np.logaddexp(0.0, scores) - y * scores)
                print(
                    f'Iter {iteration + 1:3d} | Loss: {loss:.6f} | '
                    f'||delta theta||: {update_norm:.6e}'
                )

            # Stop at the first k with ||theta_k - theta_(k-1)||_1 < eps.
            if update_norm < self.eps:
                if self.verbose:
                    print(f'Converged after {iteration + 1} iterations.')
                break

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Predicted positive-class probabilities of shape (m,).
        """
        if self.theta is None:
            raise ValueError('Model must be fitted before calling predict.')
        return self._sigmoid(np.asarray(x, dtype=float) @ self.theta)


if __name__ == '__main__':
    main(
        train_path='../data/ds1_train.csv',
        eval_path='../data/ds1_valid.csv',
        pred_path='prediction/p01b_logreg_pred_1.csv',
    )
    main(
        train_path='../data/ds2_train.csv',
        eval_path='../data/ds2_valid.csv',
        pred_path='prediction/p01b_logreg_pred_2.csv',
    )
