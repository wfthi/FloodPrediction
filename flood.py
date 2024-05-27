"""
Flood Prediction Factor, a Kaggle dataset

Walter Reade, Ashley Chow. (2024). Regression with a
Flood Prediction Dataset.
Data from Kaggle.
https://kaggle.com/competitions/playground-series-s4e5

Modified from
https://www.kaggle.com/code/aspillai/flood-prediction-regression-lightgbm-0-86931
"""
import warnings
import csv
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy import stats
from astroML.utils import split_samples
from astropy.stats import median_absolute_deviation
from astropy.stats import mad_std
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


num_cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
            'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
            'Siltation', 'AgriculturalPractices', 'Encroachments',
            'IneffectiveDisasterPreparedness', 'DrainageSystems',
            'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors']


def getFeats(df, unique_vals, mode=True, product=True):
    """
    https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499484
    https://www.kaggle.com/code/trupologhelper/ps4e5-openfe-blending-explain
    + extra features
    """
    X = df[num_cols]
    scaler = StandardScaler()

    df['ClimateAnthropogenicInteraction'] = \
        ((df['MonsoonIntensity'] + df['ClimateChange'])
         * (df['Deforestation'] + df['Urbanization']
         + df['AgriculturalPractices'] + df['Encroachments']))

    df['InfrastructurePreventionInteraction'] = \
        (df['DamsQuality'] + df['DrainageSystems'] +
         df['DeterioratingInfrastructure']) \
        * (df['RiverManagement'] + df['IneffectiveDisasterPreparedness'] +
           df['InadequatePlanning'])

    df['sum'] = X.sum(axis=1)
    Xstd = X.std(axis=1)
    Xmean = X.mean(axis=1)
    df['std'] = Xstd
    df['mean'] = Xmean
    Xmax = X.max(axis=1)
    Xmin = X.min(axis=1)
    df['max'] = Xmax
    df['min'] = Xmin
    if mode:
        print('Computing the mode - It may take a moment')
        df['mode'] = X.mode(axis=1)[0]
    else:
        print('No mode feature computed')
    df['median'] = X.median(axis=1)
    q_25th = X.quantile(0.25, axis=1)
    q_75th = X.quantile(0.75, axis=1)
    df['q_25th'] = q_25th
    df['q_75th'] = q_75th
    df['skew'] = X.skew(axis=1)
    df['kurt'] = X.kurt(axis=1)
    df['sum_72_76'] = df['sum'].isin(np.arange(72, 76))
    for i in range(10, 100, 10):
        df[f'{i}th'] = X.quantile(i / 100, axis=1)
    df['harmonic'] = stats.hmean(X, 1)
    df['geometric'] = stats.mstats.gmean(X, 1)
    df['power mean 2'] = stats.pmean(X, 2, axis=1)
    df['power mean 3'] = stats.pmean(X, 3, axis=1)
    df['power mean 4'] = stats.pmean(X, 4, axis=1)
    zcore = stats.zscore(X)
    df['zscore_sum'] = zcore.sum(1)
    df['zscore_mean'] = zcore.mean(1)
    df['zscore_median'] = np.median(zcore, 1)
    df['zscore_std'] = zcore.std(1)
    zmax = zcore.max(1)
    zmin = zcore.min(1)
    df['zscore_max'] = zmax
    df['zscore_min'] = zmin
    df['zscore_range'] = zmax - zmin
    df['cv'] = Xstd / X.mean(axis=1)
    df['Skewness_75'] = (q_75th - Xmean) / Xstd
    df['Skewness_25'] = (q_25th - Xmean) / Xstd
    df['2ndMoment'] = np.mean(X**2, 1)
    df['3rdMoment'] = np.mean(X**3, 1)
    df['entropy'] = stats.entropy(X.T)
    df['range'] = Xmax - Xmin
    df['variation'] = stats.variation(X, 1)
    df['median_absolute_deviation'] = median_absolute_deviation(X, 1)
    df['mad std'] = mad_std(X, 1)
    df['argmax'] = np.argmax(np.array(X), 1)
    df['argmin'] = np.argmin(np.array(X), 1)
    if product:
        df['product'] = np.prod(X, 1)
    for v in unique_vals:
        df['cnt_{}'.format(v)] = (df[num_cols] == v).sum(axis=1)

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


# Make predictions from trained models
def reg_predict(reg_models, X, y=None):
    ypred = []
    for i, model in enumerate(reg_models):
        if isinstance(model, list):
            ypred.append(predict_cv(X, model))
        else:
            ypred.append(model.predict(X))
        if y is not None:
            pred_r2 = r2_score(y, ypred[i])
            print('Model ', i, 'R2 score:', np.round(pred_r2, 5))
    ypred = np.array(ypred).mean(0)
    if y is not None:
        r2_cv = r2_score(y, ypred)
        print('Final test R2:', np.round(r2_cv, 5))
    return ypred


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


# Cross Validation

def predict_cv(X, reg_models):
    ypred = []
    for model in reg_models:
        ypred.append(model.predict(X))
    return np.array(ypred).mean(0)


def cross_val_train(X, y, X_test, y_test, regressor, params, spl=5):

    submission_preds, test_preds = [], []
    val_preds = np.zeros((len(X)))
    val_scores, reg_models = [], []

    cv = KFold(spl, shuffle=True, random_state=42)

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        print("Fold:", fold)
        print("Split the sample into train and validation")
        if isinstance(X, pd.core.frame.DataFrame):
            X_train = X.iloc[train_ind]
            X_val = X.iloc[valid_ind]
        else:
            X_train = X[train_ind, :]
            X_val = X[valid_ind, :]
        y_train = y[train_ind]
        y_val = y[valid_ind]

        if regressor == 'xgb':
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:  # 'lgb'
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50),
                                 lgb.log_evaluation(100)])

        reg_models.append(model)
        y_pred_trn = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        test_preds.append(y_pred_test)
        train_r2 = r2_score(y_train, y_pred_trn)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        print("Fold:", fold, " Train R2:", np.round(train_r2, 5),
              " Val R2:", np.round(val_r2, 5),
              "Test R2:", np.round(test_r2, 5))

        val_preds[valid_ind] = model.predict(X_val)
        val_scores.append(val_r2)
        print("-" * 60)

    submission_preds = np.array(submission_preds).mean(0)
    test_preds = np.array(test_preds).mean(0)
    test_r2_cv = r2_score(y_test, test_preds)
    print('Final test R2:', np.round(test_r2_cv, 5))

    return val_scores, val_preds, test_preds, reg_models


def read_data(to_pickle=False, from_pickle=False):

    if from_pickle:
        X = read_pickle('X_train.pickle')
        y = read_pickle('y_train.pickle')
        X_sub = read_pickle('X_sub.pickle')
        id_sub = read_pickle('id_sub.pickle')
        feats = read_pickle('feats.pickle')
        df_all = read_pickle('df_all.pickle')
        return X, y, X_sub, id_sub, feats, df_all

    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    print("Train:", len(df_train))
    # sample_sub = pd.read_csv("sample_submission.csv")
    df_train.head()
    print(df_train.info())

    unique_vals = []
    for df in [df_train, df_test]:
        for col in num_cols:
            unique_vals += list(df[col].unique())
    unique_vals = list(set(unique_vals))

    df_train['typ'] = 0
    df_test['typ'] = 1
    id_sub = df_test['id']
    df_all = pd.concat([df_train, df_test], axis=0)
    df_all = getFeats(df_all, unique_vals, mode=False, product=False)
    df_all.head()
    df_train = df_all[df_all['typ'] == 0]
    df_sub = df_all[df_all['typ'] == 1]

    X = df_train.drop(['id', 'FloodProbability', 'typ'], axis=1)
    y = df_train['FloodProbability']
    feats = list(X.columns)
    X_sub = df_sub[feats]

    if to_pickle:
        save_pickle(X, 'X_train.pickle')
        save_pickle(y, 'y_train.pickle')
        save_pickle(X_sub, 'X_sub.pickle')
        save_pickle(id_sub, 'id_sub.pickle')
        save_pickle(feats, 'feats.pickle')
        save_pickle(df_all, 'df_all.pickle')

    return X, y, X_sub, id_sub, feats, df_all


def train_lbg(X_train, y_train, X_test, y_test, feature_importance=False):

    # LGB model
    # 'learning_rate': 0.02
    lgb_params = {'verbosity': -1, 'n_estimators': 550, 'learning_rate': 0.015,
                  'num_leaves': 250, 'max_depth': 10, 'random_state': 0, }

    if feature_importance:
        LGB = lgb.LGBMRegressor(**lgb_params)
        LGB.fit(X, y)
        max_num_features = 50
        lgb.plot_importance(LGB, importance_type="gain",
                    figsize=(12, 8), max_num_features=max_num_features,
                    title="LightGBM Feature Importance (Gain)")
        plt.show()

    return train_models(X_train, y_train, X_test, y_test, 'lgb', lgb_params)


def train_xgb(X_train, y_train, X_test, y_test):
    xgb_params = {'n_estimators': 500,
                  'max_depth': 10,
                  'learning_rate': 0.05,
                  'eta': 0.1,  # learning rate default 0.3
                  'early_stopping_rounds': 50,
                  'eval_metric': 'rmse',
                  'seed': 202,
                  'min_child_weight': 3,
                  'verbosity': 2,
                  }
    return train_models(X_train, y_train, X_test, y_test, 'xgb', xgb_params)


def train_models(X_train, y_train, X_test, y_test,
                 regressor, params, max_depths=[8, 9, 10], spl=5):

    # Vary the max_depth to obtain different models
    reg_models_list, reg_test_preds_list = [], []
    for depth in max_depths:
        params['max_depth'] = depth
        _, _, reg_test_preds, \
            reg_models = cross_val_train(X_train,
                                         y_train,
                                         X_test, y_test,
                                         regressor,
                                         params,
                                         spl=spl)
        reg_models_list.append(reg_models)
        reg_test_preds_list.append(reg_test_preds)

    reg_test_preds_array = np.array(reg_test_preds_list)
    ypred_mean = reg_test_preds_array.mean(0)
    print('Multiple depths - Final score')
    print('test R2 mean:          ',
          r2_score(y_test, ypred_mean))
    print('test R2 median:        ',
          r2_score(y_test, np.median(reg_test_preds_array, 0)))
    print('test R2 harmonic mean: ',
          r2_score(y_test, stats.hmean(reg_test_preds_array, 0)))
    print('test R2 geometric mean:',
          r2_score(y_test, stats.mstats.gmean(reg_test_preds_array, 0)))
    return reg_models_list, ypred_mean


def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    file.close()


def read_pickle(filename):
    with open(filename, 'rb') as file:
        output = pickle.load(file)
    file.close()
    return output


if __name__ == '__main__':
    X, y, X_sub, id_sub, feats, df_all = read_data(from_pickle=True)
    (X_train, X_test), (y_train, y_test) =\
        split_samples(X, y, [0.80, 0.20], random_state=2024)
    # or train lGBM and XGBoost models
    lgb_models, lgb_test_preds = train_lbg(X_train, y_train, X_test, y_test)
    xgb_models, xgb_test_preds = train_xgb(X_train, y_train, X_test, y_test)
    test_preds = (0.7 * lgb_test_preds + 0.3 * xgb_test_preds)
    print('LGBM R2:', r2_score(y_test, lgb_test_preds))
    print('XGB R2:', r2_score(y_test, xgb_test_preds))
    print('Ensemble model R2 test:', r2_score(y_test, test_preds))
    # Save the models
    save_pickle(lgb_models, 'lgb_models.pickle')
    save_pickle(xgb_models, 'xgb_models.pickle')
    # Submission
    ypred_sub = reg_predict(lgb_models, X_sub)
    make_submission(id_sub, ypred_sub)
    #
    # from existing models
    lgb_models = read_pickle('lgb_models_0.86926.pickle')
    xgb_models = read_pickle('xgb_models_0.86926.pickle')
    ypred_lgb_train = reg_predict(lgb_models, X_train)
    ypred_lgb_test = reg_predict(xgb_models, X_test)
    save_pickle(ypred_lgb_train, 'ypred_lgb_train.pickle')
    save_pickle(ypred_lgb_test, 'ypred_xgb_test.pickle')
    plt.hist(y_train, alpha=0.5, label='train truth')
    plt.hist(ypred_lgb_train, alpha=0.5, label='train prediction')
    plt.legend()
    plt.show()