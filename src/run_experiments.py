
import os
import sys
import warnings
import json
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from xgboost import XGBClassifier, XGBRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from gaminet import GAMINet
from pygam.pygam import LogisticGAM, LinearGAM
import numpy as np
import tensorflow as tf
from datetime import datetime
# custom script imports
from src.load_datasets import load_adult_data, \
    load_telco_churn_data, load_bank_marketing_data, load_stroke_data, \
    load_airline_passenger_data, load_water_quality_data, load_fico_data, \
    load_bike_sharing_data, load_car_data, load_student_grade_data, load_crimes_data, \
    load_california_housing_data
from src.logging_reports import Accumulator

warnings.filterwarnings("ignore")


def main(model_to_run, dataset_name, task, verbose=False):
    """
    Takes model, dataset combination as input as well as classification or regression task and runs the respective
    model training and evaluation as well as calculating/averaging the metrics at the end.
    Sets the random state to 1 and n_fold for Cross Validation to 5
    Loads the respective dataset using load_datasets.py including some standardized dataset cleaning.

    :param model_to_run: model name in ['LR', 'RF', 'GBM', 'DT', 'MLP', 'XGB'] or ['GAM_Splines_traditional', 'EBM', 'Gaminet', 'EXNN', 'NAM']
    :param dataset_name: dataset name in ['water', 'stroke', 'telco', 'fico', 'bank', 'adult', 'airline'] or ['car', 'student', 'crimes', 'bike', 'housing']
    :param task: 'classification' or 'regression'
    :param verbose: True or False (default False)
    :return: None
    """

    random_state = 1  # seed
    n_folds = 5

    print('#### Run experiment on {} #####'.format(dataset_name))
    dataset, y_word_dict = None, None

    # classification data sets
    if 'water' in dataset_name:
        dataset, y_word_dict = load_water_quality_data()
    elif 'stroke' in dataset_name:
        dataset, y_word_dict = load_stroke_data()
    elif 'telco' in dataset_name:
        dataset, y_word_dict = load_telco_churn_data()
    elif 'fico' in dataset_name:
        dataset, y_word_dict = load_fico_data()
    elif 'bank' in dataset_name:
        dataset, y_word_dict = load_bank_marketing_data()
    elif 'adult' in dataset_name:
        dataset, y_word_dict = load_adult_data()
    elif 'airline' in dataset_name:
        dataset = load_airline_passenger_data()

    # Regression data sets
    elif 'car' in dataset_name:
        dataset = load_car_data()
    elif 'student' in dataset_name:
        dataset = load_student_grade_data()
    elif 'crimes' in dataset_name:
        dataset = load_crimes_data()
    elif 'bike' in dataset_name:
        dataset = load_bike_sharing_data()
    elif 'housing' in dataset_name:
        dataset = load_california_housing_data()
    else:
        raise ValueError("Dataset not in our current dataset list! Add it to the load_datasets.py procedure!")

    X = pd.DataFrame(dataset['full']['X'])
    y = np.array(dataset['full']['y'])
    X, y = shuffle(X, y, random_state=random_state)


    # This is only for fast training and experimental testing issues. This slicing is not used in the paper
    if '_1k' in dataset_name:
        X, y = X[:1000], y[:1000]

    if '_5k' in dataset_name:
        X, y = X[:5000], y[:5000]

    run_traditional_interpretable_models_and_GAMs(X, y, task,
                                                  model_to_run=model_to_run,
                                                  random_state=random_state,
                                                  dataset_name=dataset_name,
                                                  y_word_dict=y_word_dict,
                                                  n_folds=n_folds,
                                                  verbose=verbose)

    if task == "classification":
        try:
            acc = Accumulator()
            acc.log_class_report_mean_vals_with_std(model_to_run, dataset_name)
            acc.log_class_report_mean_vals(model_to_run, dataset_name)
            acc.log_timings_mean_vals(model_to_run, dataset_name)
        except Exception as e:
            print("Accumulator for Classification Reports failed, try to accumulate classification reports later\n",
                  file=sys.stderr)
            print(e, file=sys.stderr)

    elif task == "regression":
        acc = Accumulator()
        try:
            acc.log_reg_report_mean_vals_with_std(model_to_run, dataset_name)
            acc.log_timings_mean_vals(model_to_run, dataset_name)
        except Exception as e:
            print("Accumulator for Regression Reports failed, try to accumulate regression reports later\n", file=sys.stderr)
            print(e, file=sys.stderr)


def run_traditional_interpretable_models_and_GAMs(X, y, task,
                                                  model_to_run='LR',
                                                  random_state=1, dataset_name=None, y_word_dict=None, n_folds=5,
                                                  verbose=False):
    """
    Takes a Dataset represented by (X, y) as input, a task to run on (classification or regression) and the runs the model
    specified as a string from the following selection on this task:

    traditional_models_to_run = [LR', 'RF', 'GBM', 'DT', 'MLP', 'XGB']
    gam_models_to_run = ['GAM_Splines_traditional', 'EBM', 'Gaminet', 'EXNN', 'NAM']

    :param X: Train Set as pandas dataframe
    :param y: Label Set as pandas dataframe
    :param verbose: True or False for printing and logging
    :return: None

    """

    if task == "classification":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif task == "regression":
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for k_fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print('-' * 5, 'Model:', model_to_run, '-- Fold:', k_fold, '-' * 5)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        column_names = X_train.columns

        is_cat = np.array([dt.kind == 'O' for dt in X_train.dtypes])
        num_cols = X_train.columns.values[~is_cat]

        X_train = pd.get_dummies(X_train)

        # Handle unknown data as ignore
        X_test = pd.get_dummies(X_test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        dummy_column_names = X_train.columns

        num_pipe = Pipeline([('identity', FunctionTransformer()), ('scaler', StandardScaler())])
        transformers = [
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers, remainder='passthrough')

        ct.fit(X_train)
        X_train = ct.transform(X_train)
        X_test = ct.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=dummy_column_names)
        X_test = pd.DataFrame(X_test, columns=dummy_column_names)

        ############################## Traditional Approaches ##########################################################
        if 'LR' in model_to_run:
            if task == "classification":
                m2 = LogisticRegression()
                LR_start_training_time = datetime.now()
                m2.fit(X_train, y_train)
                LR_training_time = (datetime.now() - LR_start_training_time).total_seconds()
                log_timing(LR_training_time, k_fold=k_fold, dataset_name=dataset_name, model='LR')
                y_pred = m2.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='LR')
            elif task == "regression":
                m2 = Ridge()
                LR_start_training_time = datetime.now()
                m2.fit(X_train, y_train)
                LR_training_time = (datetime.now() - LR_start_training_time).total_seconds()
                log_timing(LR_training_time, k_fold=k_fold, dataset_name=dataset_name, model='LR')
                y_pred = m2.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='LR')

        if 'RF' in model_to_run:
            if task == "classification":
                m = RandomForestClassifier(n_estimators=100, max_depth=5)
                RF_start_training_time = datetime.now()
                m.fit(X_train, y_train)
                RF_training_time = (datetime.now() - RF_start_training_time).total_seconds()
                log_timing(RF_training_time, k_fold=k_fold, dataset_name=dataset_name, model='RF')
                y_pred = m.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='RF')
            elif task == "regression":
                m = RandomForestRegressor(n_estimators=100, max_depth=5)
                RF_start_training_time = datetime.now()
                m.fit(X_train, y_train)
                RF_training_time = (datetime.now() - RF_start_training_time).total_seconds()
                log_timing(RF_training_time, k_fold=k_fold, dataset_name=dataset_name, model='RF')
                y_pred = m.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='RF')

        if 'GBM' in model_to_run:
            if task == "classification":
                m5 = GradientBoostingClassifier(n_estimators=100, max_depth=5)
                GBM_start_training_time = datetime.now()
                m5.fit(X_train, y_train)
                GBM_training_time = (datetime.now() - GBM_start_training_time).total_seconds()
                log_timing(GBM_training_time, k_fold=k_fold, dataset_name=dataset_name, model='GBM')
                y_pred = m5.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='GBM')
            elif task == "regression":
                m5 = GradientBoostingRegressor(n_estimators=100, max_depth=5)
                GBM_start_training_time = datetime.now()
                m5.fit(X_train, y_train)
                GBM_training_time = (datetime.now() - GBM_start_training_time).total_seconds()
                log_timing(GBM_training_time, k_fold=k_fold, dataset_name=dataset_name, model='GBM')
                y_pred = m5.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='GBM')

        if 'DT' in model_to_run:
            if task == "classification":
                m6 = DecisionTreeClassifier(max_depth=12)
                DT_start_training_time = datetime.now()
                m6.fit(X_train, y_train)
                DT_training_time = (datetime.now() - DT_start_training_time).total_seconds()
                log_timing(DT_training_time, k_fold=k_fold, dataset_name=dataset_name, model='DT')
                y_pred = m6.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='DT')
            elif task == "regression":
                m6 = DecisionTreeRegressor(max_depth=12)
                DT_start_training_time = datetime.now()
                m6.fit(X_train, y_train)
                DT_training_time = (datetime.now() - DT_start_training_time).total_seconds()
                log_timing(DT_training_time, k_fold=k_fold, dataset_name=dataset_name, model='DT')
                y_pred = m6.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='DT')

        if 'MLP' in model_to_run:
            if task == "classification":
                mlp = MLPClassifier(max_iter=100,
                                    hidden_layer_sizes=(40),
                                    activation='relu')
                MLP_start_training_time = datetime.now()
                mlp.fit(X_train, y_train)
                MLP_training_time = (datetime.now() - MLP_start_training_time).total_seconds()
                log_timing(MLP_training_time, k_fold=k_fold, dataset_name=dataset_name, model='MLP')
                y_pred = mlp.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='MLP')
            elif task == "regression":
                mlp = MLPRegressor(max_iter=100,
                                   hidden_layer_sizes=(40),
                                   activation='relu')
                MLP_start_training_time = datetime.now()
                mlp.fit(X_train, y_train)
                MLP_training_time = (datetime.now() - MLP_start_training_time).total_seconds()
                log_timing(MLP_training_time, k_fold=k_fold, dataset_name=dataset_name, model='MLP')
                y_pred = mlp.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='MLP')

        if 'XGB' in model_to_run:
            if task == "classification":
                xgb = XGBClassifier(n_estimators=100, max_depth=5)
                XGB_start_training_time = datetime.now()
                xgb.fit(X_train, y_train)
                XGB_training_time = (datetime.now() - XGB_start_training_time).total_seconds()
                log_timing(XGB_training_time, k_fold=k_fold, dataset_name=dataset_name, model='XGB')
                y_pred = xgb.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='XGB')
            elif task == "regression":
                xgb = XGBRegressor(n_estimators=100, max_depth=5)
                XGB_start_training_time = datetime.now()
                xgb.fit(X_train, y_train)
                XGB_training_time = (datetime.now() - XGB_start_training_time).total_seconds()
                log_timing(XGB_training_time, k_fold=k_fold, dataset_name=dataset_name, model='XGB')
                y_pred = xgb.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='XGB')

        ############################## GAMs ############################################################################

        if 'GAM_Splines_traditional' in model_to_run:
            if task == "classification":
                gam = LogisticGAM()  # default: no interactions
                GAM_Splines_traditional_start_training_time = datetime.now()
                gam.fit(X_train, y_train)
                GAM_Splines_traditional_training_time = (
                        datetime.now() - GAM_Splines_traditional_start_training_time).total_seconds()
                log_timing(GAM_Splines_traditional_training_time, k_fold=k_fold, dataset_name=dataset_name,
                           model='GAM_Splines_traditional')
                y_pred = gam.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='GAM_Splines_traditional')
            elif task == "regression":
                gam = LinearGAM()  # default: no interactions
                GAM_Splines_traditional_start_training_time = datetime.now()
                gam.fit(X_train, y_train)
                GAM_Splines_traditional_training_time = (
                        datetime.now() - GAM_Splines_traditional_start_training_time).total_seconds()
                log_timing(GAM_Splines_traditional_training_time, k_fold=k_fold, dataset_name=dataset_name,
                           model='GAM_Splines_traditional')
                y_pred = gam.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='GAM_Splines_traditional')


        if 'EBM' in model_to_run:
            if task == "classification":

                m4 = ExplainableBoostingClassifier(interactions=10)
                EBM_start_training_time = datetime.now()
                m4.fit(X_train, y_train)
                EBM_training_time = (datetime.now() - EBM_start_training_time).total_seconds()
                log_timing(EBM_training_time, k_fold=k_fold, dataset_name=dataset_name, model='EBM')
                y_pred = m4.predict(X_test)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                          dataset_name=dataset_name, model='EBM')
            elif task == "regression":
                m4 = ExplainableBoostingRegressor(interactions=10)
                EBM_start_training_time = datetime.now()
                m4.fit(X_train, y_train)
                EBM_training_time = (datetime.now() - EBM_start_training_time).total_seconds()
                log_timing(EBM_training_time, k_fold=k_fold, dataset_name=dataset_name, model='EBM')
                y_pred = m4.predict(X_test)
                log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold, dataset_name=dataset_name,
                                      model='EBM')

        if 'Gaminet' in model_to_run or 'EXNN' in model_to_run:

            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            meta_info = {"X" + str(i + 1): {'type': 'continuous'} for i in range(len(X_train.columns))}
            meta_info.update({'Y': {'type': 'target'}})

            for i, (key, item) in enumerate(meta_info.items()):
                if item['type'] == 'target':
                    y_train = np.array(y_train).reshape(-1, 1)
                    y_test = np.array(y_test).reshape(-1, 1)
                else:
                    sx = MinMaxScaler((0, 1))
                    sx.fit([[0], [1]])
                    X_train = np.array(X_train)
                    X_test = np.array(X_test)
                    X_train[:, [i]] = sx.transform(np.array(X_train)[:, [i]])
                    X_test[:, [i]] = sx.transform(np.array(X_test)[:, [i]])
                    meta_info[key]['scaler'] = sx

        if 'Gaminet' in model_to_run:
            from gaminet.utils import local_visualize
            from gaminet.utils import global_visualize_density
            from gaminet.utils import feature_importance_visualize
            from gaminet.utils import plot_trajectory
            from gaminet.utils import plot_regularization

            if task == "classification":
                gami = GAMINet(meta_info=meta_info, interact_num=20,
                                       interact_arch=[40] * 5, subnet_arch=[40] * 5,
                                       batch_size=1024, task_type="Classification", activation_func=tf.nn.relu,
                                       main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                                       lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                                       heredity=True, loss_threshold=0.01, reg_clarity=1,
                                       mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                                       verbose=True, val_ratio=0.2, random_state=random_state)
                Gaminet_start_training_time = datetime.now()
                gami.fit(np.array(X_train), np.array(y_train))
                Gaminet_training_time = (datetime.now() - Gaminet_start_training_time).total_seconds()
                log_timing(Gaminet_training_time, k_fold=k_fold, dataset_name=dataset_name, model='Gaminet')
                test_pred = gami.predict(np.array(X_test))
                y_pred = np.round(test_pred)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict,
                                          k_fold=k_fold, dataset_name=dataset_name, model='Gaminet')

            elif task == "regression":
                gami = GAMINet(meta_info=meta_info, interact_num=20,
                                       interact_arch=[40] * 5, subnet_arch=[40] * 5,
                                       batch_size=1024, task_type="Regression", activation_func=tf.nn.relu,
                                       main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                                       lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                                       heredity=True, loss_threshold=0.01, reg_clarity=1,
                                       mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                                       verbose=True, val_ratio=0.2, random_state=random_state)
                Gaminet_start_training_time = datetime.now()
                gami.fit(np.array(X_train), np.array(y_train))
                Gaminet_training_time = (datetime.now() - Gaminet_start_training_time).total_seconds()
                log_timing(Gaminet_training_time, k_fold=k_fold, dataset_name=dataset_name, model='Gaminet')
                test_pred = gami.predict(np.array(X_test))
                y_pred = np.round(test_pred)
                log_regression_report(y_true=y_test, y_pred=y_pred,
                                      k_fold=k_fold, dataset_name=dataset_name, model='Gaminet')

            if verbose:
                import os
                simu_dir = "../results/"
                if not os.path.exists(simu_dir):
                    os.makedirs(simu_dir)

                # visualizations
                data_dict_logs = model_to_run.summary_logs(save_dict=False)
                plot_trajectory(data_dict_logs, folder=simu_dir, name="s1_traj_plot", log_scale=True, save_png=True)
                plot_regularization(data_dict_logs, folder=simu_dir, name="s1_regu_plot", log_scale=True, save_png=True)
                data_dict = model_to_run.global_explain(save_dict=False)
                global_visualize_density(data_dict, save_png=True, folder=simu_dir, name='s1_global')
                feature_importance_visualize(data_dict, save_png=True, folder=simu_dir, name='s1_feature')
                data_dict_local = model_to_run.local_explain(X_train[:10], y_train[:10], save_dict=False)
                local_visualize(data_dict_local[0], save_png=True, folder=simu_dir, name='s1_local')

        if 'EXNN' in model_to_run:
            from src.baseline.exnn.exnn import ExNN

            if task == "classification":
                exnn = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Classification",
                                    activation_func=tf.tanh, batch_size=min(1000, int(X_train.shape[0] * 0.2)),
                                    training_epochs=10000,  # default
                                    lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,
                                    l1_subnet=0.00316,
                                    l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
                EXNN_start_training_time = datetime.now()
                exnn.fit(X_train, y_train)
                EXNN_training_time = (datetime.now() - EXNN_start_training_time).total_seconds()
                log_timing(EXNN_training_time, k_fold=k_fold, dataset_name=dataset_name, model='EXNN')
                test_pred = exnn.predict(X_test)
                y_pred = np.round(test_pred)
                log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict,
                                          k_fold=k_fold, dataset_name=dataset_name, model='EXNN')

            elif task == "regression":
                exnn = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Regression",
                                    activation_func=tf.tanh, batch_size=min(1000, int(X_train.shape[0] * 0.2)),
                                    training_epochs=10000,  # default
                                    lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,
                                    l1_subnet=0.00316,
                                    l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
                EXNN_start_training_time = datetime.now()
                exnn.fit(X_train, y_train)
                EXNN_training_time = (datetime.now() - EXNN_start_training_time).total_seconds()
                log_timing(EXNN_training_time, k_fold=k_fold, dataset_name=dataset_name, model='EXNN')
                test_pred = exnn.predict(X_test)
                y_pred = test_pred
                log_regression_report(y_true=y_test, y_pred=y_pred,
                                      k_fold=k_fold, dataset_name=dataset_name, model='EXNN')

            if verbose:
                model_to_run.visualize("./", "exnn_demo")

        if 'NAM' in model_to_run:
            import torch
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            from pytorch_lightning.loggers import TensorBoardLogger
            from src.baseline.nam.config import defaults
            from src.baseline.nam.data import NAMDataset
            from src.baseline.nam.models import NAM, get_num_units
            from src.baseline.nam.trainer import LitNAM
            from src.baseline.nam.utils import plot_mean_feature_importance, plot_nams

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config = defaults()
            print(config)
            # https://github.com/google-research/google-research/blob/a04ff37577c2b9b337788272298c603c33859b8f/neural_additive_models/nam_train.py#L305
            config.early_stopping_patience == 60
            config.num_epochs = 1000
            config.num_basis_functions = 1000
            config.decay_rate = 0.995
            config.activation = 'exu'
            config.dropout = 0.5
            config.units_multiplier = 2
            config.optimizer = 'adam'
            config.output_regularization = 0.0
            config.feature_dropout = 0.0
            config.logdir = "../models/NAM"
            config.l2_regularization = 0.0
            config.batch_size = 1024
            config.lr = 0.01  # 0.0003

            if task == "classification":
                config.regression = False
            elif task == "regression":
                config.regression = True

            X['target'] = y
            dataset = X
            dataset = NAMDataset(config, data_path=dataset, features_columns=dataset.columns[:-1],
                                 targets_column=dataset.columns[-1])

            train_idx = train_idx.tolist()
            test_idx = test_idx.tolist()
            dataset.setup_dataloaders_custom(train_idx[0:int(len(train_idx) * 0.9)],
                                             train_idx[int(len(train_idx) * 0.9):],
                                             test_idx)
            dataloaders = dataset.train_dataloaders()

            nam_model = NAM(
                config=config,
                name="NAM",
                num_inputs=len(dataset[0][0]),
                num_units=get_num_units(config, dataset.features),
            )
            nam_model = nam_model.to(device)

            for fold, (trainloader, valloader) in enumerate(dataloaders):
                tb_logger = TensorBoardLogger(save_dir=config.logdir,
                                              name=f'{nam_model.name}',
                                              version=f'fold_{k_fold + 1}')
                checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                                      monitor='val_loss',
                                                      save_top_k=config.save_top_k,
                                                      mode='min')
                litmodel = LitNAM(config, nam_model)
                litmodel = litmodel.to(device)
                trainer = pl.Trainer(
                    logger=tb_logger,
                    max_epochs=config.num_epochs,
                    callbacks=[checkpoint_callback])  # checkpoint_callback)
                NAM_start_training_time = datetime.now()
                trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)
                NAM_training_time = (datetime.now() - NAM_start_training_time).total_seconds()
                log_timing(NAM_training_time, k_fold=k_fold, dataset_name=dataset_name, model='NAM')

            # Test trained model
            for i, data in enumerate(dataset.test_dataloaders()):
                inputs, targets = data  # get inputs
                inputs, targets = inputs.to(device), targets.to(device)
                litmodel = litmodel.to(device)
                outputs = litmodel(inputs)  # generate outputs [logits, fnns_out]
                logits = outputs[0].view(-1).tolist()
                y_test = [target[0] for target in targets.tolist()]

                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))

                test_pred = [sigmoid(logit) for logit in logits]

                if task == "classification":
                    y_pred = np.round(test_pred)
                    log_classification_report(y_true=y_test, y_pred=y_pred, y_word_dict=y_word_dict, k_fold=k_fold,
                                              dataset_name=dataset_name, model='NAM')
                elif task == "regression":
                    y_pred = test_pred
                    log_regression_report(y_true=y_test, y_pred=y_pred, k_fold=k_fold,
                                          dataset_name=dataset_name, model='NAM')


def log_classification_report(y_true, y_pred, y_word_dict, k_fold=0, dataset_name='default', model='unknown_model'):
    labels = None
    target_names = None
    if y_word_dict is not None:
        labels = []
        target_names = []
        for key_id, value_word in y_word_dict.items():
            labels.append(key_id)
            target_names.append(value_word)  # y_enc_to_word_dict[y_label_id]
    class_report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=labels,
                                                 target_names=target_names)

    classification_report_df = pd.DataFrame(class_report).transpose()
    result_dir = get_result_path_for_dataset_and_model(dataset_name, model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    classification_report_df.to_csv(f"{result_dir}/{model}_Fold_{k_fold + 1}.csv", index=True)


def log_timing(timing, k_fold=0, dataset_name='default', model='unknown_model'):
    # For current dataset gam models save log loss as csv file
    timing_df = pd.DataFrame({'model': model, 'timing': timing}, index=[k_fold + 1])

    result_dir = get_result_path_for_dataset_and_model(dataset_name, model)
    # and save
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    timing_df.to_csv(f'{result_dir}/Timing_{model}_Fold_{k_fold + 1}_.csv', float_format='%.2f', index=True)


def log_regression_report(y_true, y_pred, k_fold=0, dataset_name='default', model='unknown_model'):
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)

    regression_report_df = pd.DataFrame(np.array([[rmse, mse]]), columns=['RMSE', 'MSE'])

    result_dir = get_result_path_for_dataset_and_model(dataset_name, model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    regression_report_df.to_csv(f"{result_dir}/{model}_Fold_{k_fold + 1}.csv", index=True)


def get_result_path_for_dataset_and_model(dataset_name, model, dir='../results'):
    """
    Returns Result path for Cross Validation

    :param dataset_name:
    :param model:
    :param dir:
    :return:
    """
    return f'{dir}/{dataset_name}/{model}_Folds'


def print_results(scores):
    """
    Deprecated function to print scores simply. However, this is obsolete if logging is used as proposed.

    :param scores:
    :return:
    """
    for ds in scores.keys():
        baseline_model = list(scores[ds].keys())[0]
        try:
            # 'LR' baseline
            baseline = np.mean([s for s in scores[ds][baseline_model]])
        except:
            warnings.warn("There has to be at least one baseline model")
        print('Baseline {} - loss: {:.2f}'.format(baseline_model, baseline))
    for ds in scores.keys():
        print('', end='\t\t')
        for model in list(scores[ds].keys()):
            print("|{:^30}".format(model), end='\t\t')
        break
    print('')
    for ds in scores.keys():
        try:
            baseline = np.mean([s for s in scores[ds]['LR']])
        except:
            try:
                baseline = np.mean([s for s in scores[ds]['GAM_Splines_traditional']])
            except:
                warnings.warn("No Baseline Defined, choose at least LR or GAM Splines as Baseline")
        print(ds, end=': \t')
        for model in scores[ds].keys():
            perf = np.mean([s for s in scores[ds][model]])
            improv = (baseline - perf) / perf * 100
            print("|{:^30.2f}".format(improv), end='\t\t')
            # print('{:.2f}'.format(improv), end='\t\t')
        print('')


if __name__ == '__main__':

    traditional_models_to_run = ['LR', 'RF', 'GBM', 'DT', 'MLP', 'XGB']  # Subset or Set of: 'LR', 'RF', 'GBM', 'DT', 'MLP', 'XGB'
    gam_models_to_run = ['GAM_Splines_traditional', 'EBM', 'Gaminet', 'EXNN', 'NAM']  # Subset or Set of:  'GAM_Splines_traditional', 'EBM', 'Gaminet', 'EXNN', 'NAM'

    tasks = ['classification', 'regression']  # classification; regression
    for task in tasks:
        datasets_to_run = None
        if task == "classification":
            datasets_to_run = ['water', 'stroke', 'telco', 'fico', 'bank', 'adult', 'airline']  # Subset or Set of: ['water', 'stroke', 'telco', 'fico', 'bank', 'adult', 'airline']
        elif task == "regression":
            datasets_to_run = ['car', 'student', 'crimes', 'bike', 'housing']  # Subset or Set of: ['car', 'student', 'crimes', 'bike', 'housing']

        models = traditional_models_to_run + gam_models_to_run

        for model in models:
            for dataset in datasets_to_run:
                # Calls the wrapper function to run all models on all datasets. See details in function description
                main(model, dataset, verbose=False, task=task)

