"""
Flood Prediction Factor, a Kaggle dataset

Walter Reade, Ashley Chow. (2024). Regression with a
Flood Prediction Dataset.
Data from Kaggle.
https://kaggle.com/competitions/playground-series-s4e5

"""
import csv
# import pickle
import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from astroML.utils import split_samples
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures


def plot_correlation(x, y, xname, yname):
    plt.scatter(x, y, s=3)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()


def plot_hist(X):
    for x in np.transpose(X):
        plt.hist(x)
        plt.show()


def read_data(filename="train.csv"):
    print("Reading data from", filename)
    df = pd.read_csv(filename)
    Xname = df.columns[1:-1]
    data = np.array(df)
    X = data[:, 1:-1]
    yname = 'FloodProbability'
    y = np.array(df[yname])
    return X, Xname, y, yname


def read_test(filename="test.csv"):
    print("Reading the test data")
    df = pd.read_csv(filename)
    data = np.array(df)
    X = data[:, 1:]
    id = data[:, 0]
    Xname = df.columns[1:]
    return X, id, Xname


def sperman_rank(X, Xname, y, yname, plot_correlation=False):
    for i, (x, xn) in enumerate(zip(np.transpose(X), Xname)):
        spearmanr_coefficient, _ = spearmanr(x, y)
        print(i, xn, yname, ' - spearman R:', spearmanr_coefficient)
        if plot_correlation:
            plot_correlation(x, y, xn, yname)


def scaling(X):
    # Standard scaling of the data
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    return Xs, scaler


def decomposition_transform(X, method='PCA', n_components=None):
    if n_components is None:
        n_components = X.shape[1]
    if method == 'FastICA':
        print('FastICA transform with ', n_components, 'components')
        decomp = FastICA(n_components=n_components)
        decomp.fit(X)
    elif method == 'TruncatedSVD':
        print('TruncatedSVD transform with ', n_components, 'components')
        decomp = TruncatedSVD(n_components=n_components)
        decomp.fit(X)
    else:
        print('PCA transform with ', n_components, 'components')
        decomp = PCA(n_components=n_components)
        decomp.fit(X)
    if method != 'FastICA':
        print("Explained variance ratio", decomp.explained_variance_ratio_)
        print("Sum of explained variance ratio",
              decomp.explained_variance_ratio_.sum())

    Xdecomp = decomp.transform(X)
    #
    return Xdecomp, decomp


def transform_scale(X, pca, ica, svd,
                    pca_scaler, ica_scaler, svd_scaler):
    Xd_pca = pca.transform(X)
    Xs_pca = pca_scaler.transform(Xd_pca)
    Xd_ica = ica.transform(X)
    Xs_ica = ica_scaler.transform(Xd_ica)
    Xd_svd = svd.transform(X)
    Xs_svd = svd_scaler.transform(Xd_svd)
    return Xs_pca, Xs_ica, Xs_svd


def train_xgb(X, y, cross_validation=False,
              n_estimators=1000, max_depth=7,
              eta=0.1, n_splits=10, n_repeats=3,
              random_state=2024, eval_metric='mae',
              verbosity=2):
    print('Fitting model with XGBoost')
    # create an xgboost regression model
    # define model
    xgb_model = XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth, eta=eta,
                             eval_metric=eval_metric,
                             verbosity=verbosity)
    if cross_validation:
        print('Perform cross-validation')
        # define model evaluation method
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                           random_state=random_state)
        # evaluate model mean squared error (MAE)
        scores = cross_val_score(xgb_model, X, y,
                                 scoring='neg_mean_absolute_error',
                                 cv=cv, n_jobs=-1)
        # force scores to be positive
        scores = np.absolute(scores)
        # report a statistical summary of the performance using the mean and
        # standard deviation of the distribution of scores
        print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
    # just fit the model
    model = xgb_model.fit(X, y)
    return model


def model_predict(model, decomp, scaler, Xtest):
    Xtdecomp = decomp.transform(Xtest)
    Xts = scaler.transform(Xtdecomp)
    return model.predict(Xts)


def make_submission(id, ypred,
                    filename='flood_submission.csv'):
    fieldnames = ['id', 'FloodProbability']
    time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    fs = filename.split('.')
    file = fs[0] + '_' + time + '.csv'
    print('Output submission file', file)
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for i, y in zip(id, ypred):
            writer.writerow([i, y])
    csvfile.close()


def transform_data(X, method=None, n_components=None):
    print('Data transformation method:', method)
    Xdecomp, decomp = decomposition_transform(X, n_components=n_components,
                                              method=method)
    Xs, scaler = scaling(Xdecomp)
    return Xs, decomp, scaler


def train_multiple_xgb_models(X, y, random_states,
                              n_estimators=1000, max_depth=2,
                              eta=0.1, alpha=1.5):
    models = []
    for rs in random_states:
        print('Train XGBoost model for random state ', rs)
        (X_train, X_eval), (y_train, y_eval) =\
            split_samples(X, y, [0.75, 0.25], random_state=rs)
        xgb_model = XGBRegressor(n_estimators=n_estimators,
                                 alpha=alpha,
                                 max_depth=max_depth, eta=eta)
        model = xgb_model.fit(X_train, y_train)
        models.append(model)
        ypred_train = model.predict(X_train)
        print('Training R2 score train data:', r2_score(y_train, ypred_train))
        ypred_eval = model.predict(X_eval)
        print('Training R2 score eval data:', r2_score(y_eval, ypred_eval))
    return models


def predict_multiple_models(models, X):
    predictions = [model.predict(X) for model in models]
    return np.array(predictions)


def train(X, y, method='pca',
          max_depth=7, n_estimators=1000, cross_validation=False):

    (X_train, X_eval), (y_train, y_eval) = split_samples(X, y, [0.75, 0.25],
                                                         random_state=2024)
    Xs, decomp, scaler = transform_data(X_train, method=method)
    model = train_xgb(Xs, y_train, max_depth=max_depth,
                      n_estimators=n_estimators,
                      cross_validation=cross_validation)
    # model.get_xgb_params()
    ypred_train = model_predict(model, decomp, scaler, X_train)
    print('Training R2 score train data:', r2_score(y_train, ypred_train))
    ypred_eval = model_predict(model, decomp, scaler, X_eval)
    print('Training R2 score eval data:', r2_score(y_eval, ypred_eval))
    return model, decomp, scaler


def predict_ensemble_model(Xs_pca, Xs_ica, Xs_svd,
                           pca_xgb_models, NNregr_pca,
                           ica_xgb_models, NNregr_ica,
                           svd_xgb_models, NNregr_svd,
                           fac1=0.9):
    """
    fac1 : float between 0 and 1
    """
    pca_xgb_pred = predict_multiple_models(pca_xgb_models, Xs_pca)
    mean_pca_xgb_pred = pca_xgb_pred.mean(0)

    ica_xgb_pred = predict_multiple_models(ica_xgb_models, Xs_ica)
    mean_ica_xgb_pred = ica_xgb_pred.mean(0)

    svd_xgb_pred = predict_multiple_models(svd_xgb_models, Xs_svd)
    mean_svd_xgb_pred = svd_xgb_pred.mean(0)

    mean_xgb_pred = 0.7 * mean_ica_xgb_pred + 0.1 * mean_pca_xgb_pred +\
        0.2 * mean_svd_xgb_pred

    pca_nn_pred = NNregr_pca.predict(Xs_pca)
    ica_nn_pred = NNregr_ica.predict(Xs_ica)
    svd_nn_pred = NNregr_svd.predict(Xs_svd)

    nn_pred = 0.33 * pca_nn_pred + 0.33 * ica_nn_pred + 0.34 * svd_nn_pred

    w1 = nn_pred > mean_xgb_pred.max()
    w2 = nn_pred < mean_xgb_pred.min()
    ensemble_pred = (1. - fac1) * mean_xgb_pred + fac1 * nn_pred
    ensemble_pred[w1] = nn_pred[w1]
    ensemble_pred[w2] = nn_pred[w2]

    return pca_xgb_pred, ica_xgb_pred, svd_xgb_pred, \
        pca_nn_pred, ica_nn_pred, svd_nn_pred, nn_pred, ensemble_pred


def model_r2_score(y,
                   pca_xgb_pred,
                   ica_xgb_pred,
                   svd_xgb_pred,
                   pca_nn_pred,
                   ica_nn_pred,
                   svd_nn_pred,
                   nn_pred,
                   ensemble_pred):

    pca_xgb_models_r2 = [r2_score(y, ypred) for ypred in pca_xgb_pred]
    ica_xgb_models_r2 = [r2_score(y, ypred) for ypred in ica_xgb_pred]
    svd_xgb_models_r2 = [r2_score(y, ypred) for ypred in svd_xgb_pred]

    print('R2 score')
    print('XGBoost PCA models R2:', pca_xgb_models_r2)
    print('XGBoost ICA models R2:', ica_xgb_models_r2)
    print('XGBoost SVD models R2:', svd_xgb_models_r2)
    print('NN PCA model :', r2_score(y, pca_nn_pred))
    print('NN ICA model :', r2_score(y, ica_nn_pred))
    print('NN SVD model :', r2_score(y, svd_nn_pred))
    print('average NN model :', r2_score(y, nn_pred))
    print('Ensemble model :', r2_score(y, ensemble_pred))


def train_ensemble_model(Xs_pca, Xs_ica, Xs_svd, y,
                         random_states=[2024, 12, 54, 2414, 484]):
    # train muliptle XGB models
    print()
    print('Train XGbosst with after pca transform')
    pca_xgb_models = train_multiple_xgb_models(Xs_pca, y, random_states,
                                               n_estimators=800, max_depth=5)

    print()
    print('Train XGbosst with after ica transform')
    ica_xgb_models = train_multiple_xgb_models(Xs_ica, y, random_states,
                                               n_estimators=800, max_depth=5)

    print()
    print('Train XGbosst with after svd transform')
    svd_xgb_models = train_multiple_xgb_models(Xs_svd, y, random_states,
                                               n_estimators=800, max_depth=5)

    # Train neural networks
    learning_rate = 'adaptive'
    hidden_layer_sizes = [10, 10, 10]
    print()
    print('Train Neural Network with after pca transform')
    NNregr_pca = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                              random_state=1, max_iter=500, verbose=True,
                              early_stopping=True, validation_fraction=0.25,
                              learning_rate=learning_rate,
                              ).fit(Xs_pca, y)

    print()
    print('Train Neural Network with after ica transform')
    NNregr_ica = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                              random_state=1, max_iter=500, verbose=True,
                              early_stopping=True, validation_fraction=0.25,
                              learning_rate=learning_rate,
                              ).fit(Xs_ica, y)

    print()
    print('Train Neural Network with after svd transform')
    NNregr_svd = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                              random_state=1, max_iter=500, verbose=True,
                              early_stopping=True, validation_fraction=0.25,
                              learning_rate=learning_rate,
                              ).fit(Xs_svd, y)

    return pca_xgb_models, NNregr_pca, ica_xgb_models, NNregr_ica, \
        svd_xgb_models, NNregr_svd


def feature_engineering(X, Xname):
    Xname = list(Xname)
    _, Xminmax, Xmean, Xvar, Xskewness, Xkurtosis =\
        stats.describe(X, axis=1)
    Xl = list(np.transpose(X))
    Xl.append(Xmean)
    Xname.append('Mean')
    Xl.append(np.median(X, 1))
    Xname.append('Median')
    Xl.append(np.sqrt(Xvar))
    Xname.append('Std')
    Xl.append(Xminmax[0])
    Xname.append('Xmin')
    Xl.append(Xminmax[1])
    Xname.append('Xmax')
    Xl.append(np.mean(X**2, 1))
    Xname.append('<x**2>')
    Xl.append(np.mean(X**3, 1))
    Xname.append('<x**3>')

    # --- Not so useful engineered features
    Xl.append(np.sum(X, 1))
    Xname.append('sum')  # sum = 20 * mean
    Xl.append(Xskewness)
    Xname.append('Xskewness')
    Xl.append(Xkurtosis)
    Xname.append('Xkurtosis')
    Xl.append(np.argmax(X, 1))
    Xname.append('argmax')
    Xl.append(np.argmin(X, 1))
    Xname.append('argmin')
    for ir in range(5):
        Xr = Xl - np.roll(Xl, ir + 1)
        Xl.append(Xr[0, :])
        fname = 'shift_' + str(ir + 1)
        Xname.append(fname)
    X = np.transpose(Xl)
    return X, np.array(Xname)


if __name__ == "__main__":
    Xinp, Xname_inp, y, yname = read_data()
    Xf, Xfname = feature_engineering(Xinp, Xname_inp)
    sperman_rank(Xf, Xfname, y, yname)
    plt.scatter(Xf[:, 20], y, s=1)
    plt.scatter(Xf[:, 26], y, s=1)
    plt.show()
    X = Xf[:, 20:25]
    Xname = Xfname[20:25]
    poly = PolynomialFeatures(3)
    Xp = poly.fit_transform(X)
    Xp = Xp[:, 1:]
    Xname_p = ['p' + str(i) for i in range(Xp.shape[1] - X.shape[1])]
    Xname_p = Xname_p[1:]
    sperman_rank(Xp, Xname_p, y, yname)
    X = Xp
    (X_train, X_eval), (y_train, y_eval) =\
        split_samples(X, y, [0.75, 0.25], random_state=2024)
    n_components = X.shape[1]
    n_components = 20
    Xs_pca, pca, pca_scaler = transform_data(X_train, method='PCA',
                                             n_components=n_components)
    Xs_ica, ica, ica_scaler = transform_data(X_train, method='FastICA',
                                             n_components=n_components)
    Xs_svd, svd, svd_scaler = transform_data(X_train, method='TruncatedSVD',
                                             n_components=n_components)

    pca_xgb_models, NNregr_pca, ica_xgb_models, NNregr_ica, \
        svd_xgb_models, NNregr_svd =\
        train_ensemble_model(Xs_pca, Xs_ica, Xs_svd, y_train)

    pca_xgb_pred, ica_xgb_pred, svd_xgb_pred, \
        pca_nn_pred, ica_nn_pred, svd_nn_pred, nn_pred, ensemble_pred = \
        predict_ensemble_model(Xs_pca, Xs_ica, Xs_svd,
                               pca_xgb_models, NNregr_pca,
                               ica_xgb_models, NNregr_ica,
                               svd_xgb_models, NNregr_svd)

    model_r2_score(y_train,
                   pca_xgb_pred,
                   ica_xgb_pred,
                   svd_xgb_pred,
                   pca_nn_pred,
                   ica_nn_pred,
                   svd_nn_pred,
                   nn_pred,
                   ensemble_pred)
    #

    Xs_eval_pca, Xs_eval_ica, Xs_eval_svd = transform_scale(X_eval,
                                                            pca, ica, svd,
                                                            pca_scaler,
                                                            ica_scaler,
                                                            svd_scaler)

    pca_xgb_eval_pred, ica_xgb_eval_pred, svd_xgb_eval_pred, \
        pca_nn_eval_pred, ica_nn_eval_pred, svd_nn_eval_pred, \
        nn_eval_pred, ensemble_eval_pred = \
        predict_ensemble_model(Xs_eval_pca, Xs_eval_ica, Xs_eval_svd,
                               pca_xgb_models, NNregr_pca,
                               ica_xgb_models, NNregr_ica,
                               svd_xgb_models, NNregr_svd)

    model_r2_score(y_eval,
                   pca_xgb_eval_pred,
                   ica_xgb_eval_pred,
                   svd_xgb_eval_pred,
                   pca_nn_eval_pred,
                   ica_nn_eval_pred,
                   svd_nn_eval_pred,
                   nn_eval_pred,
                   ensemble_eval_pred)

    # Learn with all the Data for test submission
    X_train = X
    y_train = y
    Xs_pca, pca, pca_scaler = transform_data(X_train, method='PCA',
                                             n_components=n_components)
    Xs_ica, ica, ica_scaler = transform_data(X_train, method='FastICA',
                                             n_components=n_components)
    Xs_svd, svd, svd_scaler = transform_data(X_train, method='TruncatedSVD',
                                             n_components=n_components)
    pca_xgb_models, NNregr_pca, ica_xgb_models, NNregr_ica, \
        svd_xgb_models, NNregr_svd =\
        train_ensemble_model(Xs_pca, Xs_ica, Xs_svd, y_train)
    #
    X_test, id, _ = read_test()
    X_test, Xfname = feature_engineering(X_test, Xname_inp)
    Xp = poly.transform(X_test[:, 20:25])
    X_test = Xp[:, 1:]
    Xs_test_pca, Xs_test_ica, Xs_test_svd = transform_scale(X_test,
                                                            pca, ica, svd,
                                                            pca_scaler,
                                                            ica_scaler,
                                                            svd_scaler)
    _, _, _, _, _, _, _, ensemble_pred_test =\
        predict_ensemble_model(Xs_test_pca, Xs_test_ica, Xs_test_svd,
                               pca_xgb_models, NNregr_pca,
                               ica_xgb_models, NNregr_ica,
                               svd_xgb_models, NNregr_svd)
    make_submission(id, ensemble_pred_test)
    # -----------------------------------------------------------------

