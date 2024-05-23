"""

https://www.kaggle.com/code/aspillai/flood-prediction-regression-lightgbm-0-86931
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
# from scipy.stats import spearmanr
from scipy import stats
from astroML.utils import split_samples
from astropy.stats import median_absolute_deviation
from astropy.stats import mad_std
from xgboost.sklearn import XGBRegressor

warnings.filterwarnings("ignore")


num_cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
            'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
            'Siltation', 'AgriculturalPractices', 'Encroachments',
            'IneffectiveDisasterPreparedness', 'DrainageSystems',
            'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors']


def getFeats(df, mode=True):
    """
    https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499484
    https://www.kaggle.com/code/trupologhelper/ps4e5-openfe-blending-explain
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
    # df['product'] = np.prod(X, 1)
    for v in unique_vals:
        df['cnt_{}'.format(v)] = (df[num_cols] == v).sum(axis=1)

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


# Make predictions from trained models
def reg_predict(reg_models, X, y=None):
    ypred = []
    for i, model in enumerate(reg_models):
        ypred.append(model.predict(X))
        if y is not None:
            pred_r2 = r2_score(y, ypred[i])
            print('Model ', i, 'R2 score:', np.round(pred_r2, 5))
    ypred = np.array(ypred).mean(0)
    r2_cv = r2_score(y, ypred)
    print('Final test R2:', np.round(r2_cv, 5))
    return ypred


# Cross Validation
def cross_val_train(X, y, X_test, y_test, df_sub,
                    feats, params, regressor='lgb', spl=5):

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

        submission_preds.append(model.predict(df_sub[feats]))
        val_preds[valid_ind] = model.predict(X_val)
        val_scores.append(val_r2)
        print("-" * 60)

    submission_preds = np.array(submission_preds).mean(0)
    test_preds = np.array(test_preds).mean(0)
    test_r2_cv = r2_score(y_test, test_preds)
    print('Final test R2:', np.round(test_r2_cv, 5))

    return val_scores, val_preds, test_preds, submission_preds, reg_models


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
df_all = pd.concat([df_train, df_test], axis=0)
df_all = getFeats(df_all, mode=False)
df_all.head()
df_train = df_all[df_all['typ'] == 0]
df_sub = df_all[df_all['typ'] == 1]

X = df_train.drop(['id', 'FloodProbability', 'typ'], axis=1)
y = df_train['FloodProbability']
feats = list(X.columns)

(X_train, X_test), (y_train, y_test) =\
            split_samples(X, y, [0.80, 0.20], random_state=2024)

# LGB model
params = {'verbosity': -1, 'n_estimators': 550, 'learning_rate': 0.02,
          'num_leaves': 250, 'max_depth': 10, 'random_state': 0, }
LGB = lgb.LGBMRegressor(**params)
LGB.fit(X, y)
max_num_features = 50
lgb.plot_importance(LGB, importance_type="gain",
                    figsize=(12, 8), max_num_features=max_num_features,
                    title="LightGBM Feature Importance (Gain)")
plt.show()

lgb_val_scores, lgb_val_preds, lgb_test_preds, \
    lgb_submission_preds, lgb_models = cross_val_train(X_train,
                                                       y_train,
                                                       X_test, y_test,
                                                       df_sub,
                                                       feats,
                                                       params,
                                                       spl=5)

# Evaluate the model
mse = mean_squared_error(y_train, lgb_val_preds)
rmse = np.sqrt(mean_squared_error(y_train, lgb_val_preds))
r2 = r2_score(y_train, lgb_val_preds)
#
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Final test R2: 0.86913
# Final test R2: 0.86914
# Final test R2: 0.86913
# Final test R2: 0.86913
# Final test R2: 0.86848 if split_samples 0.90/0.10

# XGBoost model
params = {'n_estimators': 500,
          'max_depth': 10,
          'learning_rate': 0.05,
          'eta': 0.1,  # learning rate default 0.3
          'early_stopping_rounds': 50,
          'eval_metric': 'rmse',
          'seed': 202,
          'min_child_weight': 3,
          'verbosity': 2,
          }
xgb_val_scores, xgb_val_preds, xgb_test_preds, xgb_submission_preds, \
    xgb_models = \
    cross_val_train(X_train,
                    y_train,
                    X_test, y_test,
                    df_sub,
                    feats,
                    params,
                    regressor='xgb',
                    spl=5)
# Final test R2: 0.86898
# Final test R2: 0.86897
# Final test R2: 0.869
# Final test R2: 0.86897
# Final test R2: 0.86903
test_preds = (0.7 * lgb_test_preds + 0.3 * xgb_test_preds)
