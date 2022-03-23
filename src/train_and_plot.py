# %%

import os
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from load_datasets import load_water_quality_data, load_bank_marketing_data, load_adult_data, load_bike_sharing_data, \
    load_stroke_data, load_telco_churn_data, load_fico_data, load_adult_data, load_bank_marketing_data, \
    load_airline_passenger_data, load_car_data, load_student_grade_data, load_crimes_data, load_bike_sharing_data, \
    load_california_housing_data
from pygam.pygam import LogisticGAM, LinearGAM
from baseline.exnn.exnn.exnn import ExNN
import tensorflow as tf
from gaminet import GAMINet

import torch
import pytorch_lightning as pl
from baseline.nam.config import defaults
from baseline.nam.data import NAMDataset
from baseline.nam.models import NAM, get_num_units
from baseline.nam.trainer import LitNAM
from pytorch_lightning.loggers import TensorBoardLogger

random_state = 1
task = 'classification'  # regression or classification

dataset, _ = load_adult_data()
dataset_name = 'adult'

X = pd.DataFrame(dataset['full']['X'])
y = np.array(dataset['full']['y'])
X, y = shuffle(X, y, random_state=random_state)

is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])

num_cols = X.columns.values[~is_cat]

# Handle unknown data as ignore
X = pd.get_dummies(X)
X = X.reindex(columns=X.columns, fill_value=0)
dummy_column_names = X.columns

num_pipe = Pipeline([('identity', FunctionTransformer())])
transformers = [
    ('num', num_pipe, num_cols)
]
ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
ct.fit(X)
X = ct.transform(X)

X = pd.DataFrame(X, columns=dummy_column_names)

scaler_dict = {}
for c in num_cols:
    # Use MinMaxScaler for Gaminet since it scales back the continuous features with this scaler internally
    scaler = MinMaxScaler()
    # Use RobustScaler with other models that can be scalled back manually
    # scaler = RobustScaler()
    X[c] = scaler.fit_transform(X[c].values.reshape(-1, 1))
    scaler_dict[c] = scaler


def feature_importance_visualize(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):
    """
    Visualizes feature importances for EBM and Gaminet

    :param data_dict_global:
    :param folder:
    :param name:
    :param save_png:
    :param save_eps:
    :return:
    """
    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.6 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1])
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=60)
        plt.xlabel("Feature Name", fontsize=12)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Feature Importance")

        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
    plt.show()


# %%

def make_plot(x, mean, upper_bounds, lower_bounds, feature_name, model_name, dataset_name, scale_back=True):
    """
    Scales back continuous features if wanted (default True) and makes a line plot

    :param x:
    :param mean:
    :param upper_bounds:
    :param lower_bounds:
    :param feature_name:
    :param model_name:
    :param dataset_name:
    :param scale_back:
    :return:
    """
    x = np.array(x)
    if feature_name in num_cols and scale_back:
        x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
    plt.plot(x, mean, color='black')
    plt.fill_between(x, lower_bounds, mean, color='gray')
    plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'{model_name} - Feature:{feature_name}')
    plt.savefig(f'../plots/{model_name}_{dataset_name}_shape_{feature_name}.pdf')
    plt.show()


def make_plot_ebm(data_dict, feature_name, model_name, dataset_name):
    """
    Plots a shape plot as a step function explicitly, since EBM at InterpretMl is designed to learn step functions

    :param data_dict:
    :param feature_name:
    :param model_name:
    :param dataset_name:
    :param num_epochs:
    :param debug:
    :return:
    """
    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()

    # This is important since you do not plot plt.stairs with len(edges) == len(vals) + 1, which will have a drop to zero at the end
    y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]

    # This is the code interpretml also uses:
    # https://github.com/interpretml/interpret/blob/2327384678bd365b2c22e014f8591e6ea656263a/python/interpret-core/interpret/visual/plot.py#L115
    # This is our custom code used for plotting

    x = np.array(x_vals)

    if feature_name in num_cols:
        x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()

    plt.step(x, y_vals, where="post", color='black')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}.pdf')
    plt.show()


def make_plot_interaction(left_names, right_names, scores, feature_name_left, feature_name_right, model_name,
                          dataset_name, scale_back=True):
    """
    Makes a matplotlib heatmap interaction plot. If categorical features are included these are interpreted as strings
    instead of floats (0.0 and 1.0) or ints. Continuous values are scaled back if wanted.

    :param left_names:
    :param right_names:
    :param scores:
    :param feature_name_left:
    :param feature_name_right:
    :param model_name:
    :param dataset_name:
    :param scale_back:
    :return:
    """

    left_names = np.array(left_names)

    if feature_name_left in num_cols and scale_back:
        left_names = scaler_dict[feature_name_left].inverse_transform(left_names.reshape(-1, 1)).squeeze()
    if "_" in feature_name_left:
        left_names = left_names.astype('str')
    right_names = np.array(right_names)
    if feature_name_right in num_cols and scale_back:
        right_names = scaler_dict[feature_name_right].inverse_transform(right_names.reshape(-1, 1)).squeeze()
    if "_" in feature_name_right:
        right_names = right_names.astype('str')
    fig, ax = plt.subplots()
    im = ax.pcolormesh(left_names, right_names, scores, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.savefig(
        f'../plots/{model_name}_{dataset_name}_interact_{feature_name_left.replace("?", "missing")}x{feature_name_right.replace("?", "missing")}.png')
    plt.show()


def make_plot_interaction_continuous_x_cat_ebm(left_names, right_names, scores, feature_name_left, feature_name_right,
                                               model_name,
                                               dataset_name):
    """
    Makes matplotlib interaction plots especially for ebm with continuous x categorical variables.

    :param left_names:
    :param right_names:
    :param scores:
    :param feature_name_left:
    :param feature_name_right:
    :param model_name:
    :param dataset_name:
    :return:
    """

    left_names = np.array(left_names)

    if feature_name_left in num_cols:
        left_names = scaler_dict[feature_name_left].inverse_transform(left_names.reshape(-1, 1)).squeeze()
    right_names = np.array(right_names)
    if feature_name_right in num_cols:
        right_names = scaler_dict[feature_name_right].inverse_transform(right_names.reshape(-1, 1)).squeeze()

    scores = np.r_[scores, scores[np.newaxis, -1]]
    scores = np.transpose(scores)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(left_names, right_names, scores, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.savefig(
        f'plots/{model_name}_{dataset_name}_interact_{feature_name_left.replace("?", "")} x {feature_name_right.replace("?", "")}.pdf')
    plt.show()


def make_one_hot_plot(class_zero, class_one, feature_name, model_name, dataset_name):
    """
    Makes one hot plot

    :param class_zero:
    :param class_one:
    :param feature_name:
    :param model_name:
    :param dataset_name:
    :param num_epochs:
    :return:
    """

    plt.bar([0, 1], [class_zero, class_one], color='gray', tick_label=[f'{feature_name} = 0', f'{feature_name} = 1'])
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    plt.savefig(f'../plots/{model_name}_{dataset_name}_onehot_{feature_name.replace("?", "missing")}.pdf')
    plt.show()


# %%

def EBM_show(X, y):
    """
    Uses the lib function from InterpretML to show the EBM plotting dashboard. This is probably the fastes way to explore

    :param X:
    :param y:
    :return:
    """
    m4 = ExplainableBoostingRegressor(interactions=10, max_bins=256)
    m4.fit(X, y)
    ebm_global = m4.explain_global()
    show(ebm_global)


def EBM(X, y, dataset_name, model_name='EBM'):
    """
    Trains and plots EBM shape and interaction functions

    :param X:
    :param y:
    :param dataset_name:
    :param model_name:
    :return:
    """
    if task == "classification":
        ebm = ExplainableBoostingClassifier(interactions=10, max_bins=256)
    else:
        ebm = ExplainableBoostingRegressor(interactions=10, max_bins=256)
    ebm.fit(X, y)
    ebm_global = ebm.explain_global()
    for i in range(len(ebm_global.data()['names'])):
        data_names = ebm_global.data()
        feature_name = data_names['names'][i]
        shape_data = ebm_global.data(i)
        if shape_data['type'] == 'interaction':
            x_name, y_name = feature_name.split(' x ')
            if len(shape_data['left_names']) > 2 and len(shape_data['right_names']) == 2:
                make_plot_interaction_continuous_x_cat_ebm(shape_data['left_names'], shape_data['right_names'],
                                                           shape_data['scores'],
                                                           x_name, y_name, model_name, dataset_name)
            else:
                make_plot_interaction(shape_data['left_names'], shape_data['right_names'],
                                      np.transpose(shape_data['scores']),
                                      x_name, y_name, model_name, dataset_name)
            continue
        if len(shape_data['names']) == 2:
            make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
        else:
            make_plot_ebm(shape_data, feature_name, model_name, dataset_name)

    feat_for_vis = dict()
    for i, n in enumerate(ebm_global.data()['names']):
        feat_for_vis[n] = {'importance': ebm_global.data()['scores'][i]}
    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='ebm_feat_imp')


def GAM(X, y, dataset_name, model_name='GAM'):
    """
    Trains and plots GAM Splines shape and interaction functions.

    :param X:
    :param y:
    :param dataset_name:
    :param model_name:
    :return:
    """
    if task == "classification":
        gam = LogisticGAM().fit(X, y)
    elif task == "regression":
        gam = LinearGAM().fit(X, y)
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        if len(X[X.columns[i]].unique()) == 2:
            make_one_hot_plot(pdep[0], pdep[-1], X.columns[i], model_name, dataset_name)
        else:
            make_plot(XX[:, i].squeeze(), pdep, pdep, pdep, X.columns[i], model_name, dataset_name)


def Gaminet(X, y, dataset_name, model_name='Gaminet'):
    """
    Trains and plots Gaminet shape and interaction functions

    :param X:
    :param y:
    :param dataset_name:
    :param model_name:
    :return:
    """
    x_types = {}
    for i in range(len(X.columns)):
        if "_" in X.columns[i]:
            x_types[X.columns[i]] = {'type': 'categorical', "values": [0, 1]}
        else:
            x_types[X.columns[i]] = {'type': 'continuous'}

    meta_info = x_types
    meta_info.update({'Y': {'type': 'target'}})

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            continue
        if key in scaler_dict and item['type'] != 'categorical':
            meta_info[key]['scaler'] = scaler_dict[key]
        else:
            identity = FunctionTransformer()
            meta_info[key]['scaler'] = identity

    if task == "classification":
        # Modify amount of interactions in case you want to extend to further interesting interaction plots
        model_to_run = GAMINet(meta_info=meta_info, interact_num=10,
                               interact_arch=[40] * 5, subnet_arch=[40] * 5,
                               batch_size=1024, task_type="Classification", activation_func=tf.nn.relu,
                               main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                               lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=True, loss_threshold=0.01, reg_clarity=1,
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        print(np.array(y).shape)
        model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

    elif task == "regression":
        # Modify amount of interactions in case you want to extend to further interesting interaction plots
        model_to_run = GAMINet(meta_info=meta_info, interact_num=10,
                               interact_arch=[40] * 5, subnet_arch=[40] * 5,
                               batch_size=1024, task_type="Regression", activation_func=tf.nn.relu,
                               main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                               lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=True, loss_threshold=0.01, reg_clarity=1,
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        model_to_run.fit(np.array(X), np.array(y))

    data_dict = model_to_run.global_explain(save_dict=False, main_grid_size=256)

    Xnames2Featurenames = dict(zip([X.columns[i] for i in range(X.shape[1])], X.columns))
    print(Xnames2Featurenames)

    for k in data_dict.keys():
        if data_dict[k]['type'] == 'pairwise':
            feature_name_left, feature_name_right = k.split(' vs. ')

            make_plot_interaction(data_dict[k]['input1'], data_dict[k]['input2'], data_dict[k]['outputs'],
                                  feature_name_left,
                                  feature_name_right,
                                  model_name, dataset_name, scale_back=False)


        elif data_dict[k]['type'] == 'continuous':
            make_plot(data_dict[k]['inputs'], data_dict[k]['outputs'], data_dict[k]['outputs'],
                      data_dict[k]['outputs'], Xnames2Featurenames[k], model_name, dataset_name, scale_back=False)

        elif data_dict[k]['type'] == 'categorical':  ####len(X[Xnames2Featurenames[k]].unique()) == 2:
            make_one_hot_plot(data_dict[k]['outputs'][0], data_dict[k]['outputs'][-1],
                              Xnames2Featurenames[k], model_name, dataset_name)
        else:
            continue

    feat_for_vis = dict()
    for i, k in enumerate(data_dict.keys()):
        if 'vs.' in k:
            feature_name_left, feature_name_right = k.split(' vs. ')

            feat_for_vis[f'{feature_name_left}\nvs.\n{feature_name_right}'] = {'importance': data_dict[k]['importance']}
        else:
            feat_for_vis[Xnames2Featurenames[k]] = {'importance': data_dict[k]['importance']}

    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='gaminet_feat_imp')


def EXNN(X, y, dataset_name, model_name='ExNN'):
    """
    Trains and plots ExNN shape functions

    :param X:
    :param y:
    :param dataset_name:
    :param model_name:
    :return:
    """

    meta_info = {"X" + str(i + 1): {'type': 'continuous'} for i in range(len(X.columns))}
    meta_info.update({'Y': {'type': 'target'}})

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            continue
        sx = MinMaxScaler((0, 1))
        sx.fit([[0], [1]])
        meta_info[key]['scaler'] = sx

    X_arr = np.array(X)
    y_arr = np.array(y)

    if task == "classification":
        model_to_run = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Classification",
                            activation_func=tf.tanh, batch_size=min(1000, int(X.shape[0] * 0.2)),
                            training_epochs=10000,  # default 10000
                            lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,

                            l1_subnet=0.00316,
                            l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(save_png=True, folder='../plots/', name=f'{model_name}_{dataset_name}_shape')

    elif task == "regression":
        model_to_run = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Regression",
                            activation_func=tf.tanh, batch_size=min(1000, int(X.shape[0] * 0.2)),
                            training_epochs=10000,  # default
                            lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,
                            l1_subnet=0.00316,
                            l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(save_png=True, folder='../plots/', name=f'{model_name}_{dataset_name}_shape')


def NAM_runner(X, y, dataset_name, model_name='NAM'):
    """
    Trains and plots NAM shape functions

    :param X:
    :param y:
    :param dataset_name:
    :param model_name:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = defaults()
    config.early_stopping_patience == 60
    config.num_epochs = 1000
    config.num_basis_functions = 1000
    config.decay_rate = 0.995
    config.activation = 'relu'  # 'exu'
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
    elif task == "cegression":
        config.regression = True

    X['target'] = y
    dataset = X
    dataset = NAMDataset(config, data_path=dataset, features_columns=dataset.columns[:-1],
                         targets_column=dataset.columns[-1])

    train_idx = np.arange(len(X)).tolist()
    test_idx = np.arange(len(X)).tolist()
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
        litmodel = LitNAM(config, nam_model)
        litmodel = litmodel.to(device)
        tb_logger = TensorBoardLogger(save_dir="../models/NAM_Plot",
                                      name=f'{nam_model.name}')

        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=config.save_top_k,
                                              mode='min')

        trainer = pl.Trainer(
            max_epochs=config.num_epochs,
            callbacks=[checkpoint_callback])

        trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)

    for i, feature_name in enumerate(X.drop('target', axis=1).columns):
        inp = torch.linspace(X[feature_name].min(), X[feature_name].max(), 1000)
        outp = nam_model.feature_nns[i](inp).detach().numpy().squeeze()
        if len(X[feature_name].unique()) == 2:
            make_one_hot_plot(outp[0], outp[-1], feature_name, model_name, dataset_name, config.num_epochs)
        else:
            make_plot(inp, outp, outp, outp, feature_name, model_name, dataset_name, config.num_epochs)


# %%
# MAIN These lines are made for plotting each model. The comments show jupyter notebook cells so that this file can be
# easily copied to a jupyter notebook and esp. EBM_show can be used within a jupyter notebook easily.

# EBM_show(X, y)

# %%

# MAIN These lines are made for plotting each model. Uncomment the model wished to train and plot. See details at
# Scaling in the very beginning of this script and in the respective Model call here for amount of interactions and
# epochs if present.

EBM(X, y, dataset_name)
# GAM(X, y, dataset_name)
# Gaminet(X, y, dataset_name)
# EXNN(X, y, dataset_name)
# NAM_runner(X, y, dataset_name)
