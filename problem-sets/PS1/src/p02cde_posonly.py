import numpy as np
import util
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, t_train)

    # Plot data and decision boundary next to the prediction file.
    plot_path = '{}.png'.format(pred_path_c.rsplit('.', 1)[0])
    util.plot(x_train, t_train, model.theta, plot_path)

    # Save predictions
    x_eval, _ = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    predictions = model.predict(x_eval)
    np.savetxt(pred_path_c, predictions >= 0.5, fmt='%d')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary next to the prediction file.
    plot_path = '{}.png'.format(pred_path_d.rsplit('.', 1)[0])
    util.plot(x_train, y_train, model.theta, plot_path)

    # Save predictions
    x_eval, _ = util.load_dataset(valid_path, add_intercept=True)
    predictions = model.predict(x_eval)
    np.savetxt(pred_path_d, predictions >= 0.5, fmt='%d')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    alpha = model.predict(x_valid[y_valid == 1]).mean()

    # Corrected boundary solves P(y=1|x)/alpha = 1/2, i.e. theta'^T x = 0
    theta_prime = model.theta + np.log(2 / alpha - 1) * np.array([1, 0, 0])
    plot_path = '{}.png'.format(pred_path_e.rsplit('.', 1)[0])
    util.plot(x_train, y_train, theta_prime, plot_path)

    # Save predictions
    x_eval, _ = util.load_dataset(valid_path, add_intercept=True)
    predictions = model.predict(x_eval) / alpha
    np.savetxt(pred_path_e, predictions >= 0.5, fmt='%d')
    # *** END CODE HERE ***
