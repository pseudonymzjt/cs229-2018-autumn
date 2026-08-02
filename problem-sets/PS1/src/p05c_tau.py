import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(c): Tune the bandwidth parameter tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load datasets
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    def validation_mse(tau):
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_valid)
        return np.mean((y_pred - y_valid) ** 2)

    # Instead of committing to the fixed tau_values grid, use it only to bound
    # the search, then refine the minimizer with golden-section search on
    # log(tau). Validation MSE is roughly unimodal in log(tau): tiny tau
    # overfits, huge tau underfits.
    lo, hi = np.log(min(tau_values)), np.log(max(tau_values))
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - golden_ratio * (hi - lo)
    d = lo + golden_ratio * (hi - lo)
    mse_c = validation_mse(np.exp(c))
    mse_d = validation_mse(np.exp(d))

    while hi - lo > 1e-3:
        if mse_c < mse_d:
            hi, d, mse_d = d, c, mse_c
            c = hi - golden_ratio * (hi - lo)
            mse_c = validation_mse(np.exp(c))
        else:
            lo, c, mse_c = c, d, mse_d
            d = lo + golden_ratio * (hi - lo)
            mse_d = validation_mse(np.exp(d))

    best_tau = np.exp((lo + hi) / 2.0)
    best_mse = validation_mse(best_tau)

    # Safety net: keep the best fixed-grid candidate if it beats the
    # refinement, so the result is never worse than a plain grid search.
    for tau in tau_values:
        mse_tau = validation_mse(tau)
        if mse_tau < best_mse:
            best_tau, best_mse = tau, mse_tau

    print('Coarse grid MSEs: {}'.format(
        {tau: round(validation_mse(tau), 6) for tau in tau_values}))
    print('Best tau: {:.6f} | validation MSE: {:.6f}'.format(best_tau, best_mse))

    # Fit LWR with the best tau, run on the test set, and save predictions
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print('Test MSE (tau={:.6f}): {:.6f}'.format(best_tau, test_mse))
    np.savetxt(pred_path, y_test_pred, fmt='%.6f')

    # Plot data
    plot_path = '{}.png'.format(pred_path.rsplit('.', 1)[0])
    plt.figure()
    plt.plot(x_test[:, -1], y_test, 'bx', label='label')
    plt.plot(x_test[:, -1], y_test_pred, 'ro', label='prediction')
    plt.suptitle('Test Set (tau={:.4f}, MSE={:.6f})'.format(best_tau, test_mse),
                 fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig(plot_path)
    # *** END CODE HERE ***
