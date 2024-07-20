"""
Created on Thu May 16 15:29:51 2024
@author: raza0005
"""

# %% Import libraries and function definition
# %%% Relevant libraries

import os
import csv
import sys
import math
import torch
import pickle
import random
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from pandas import read_csv
from pycaret.regression import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, explained_variance_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from packaging import version
from ctgan import CTGAN
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

from sdv.sampling import Condition
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns

from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%% Functions definations

def dBm2mW(dBm_val):
    return 10**((dBm_val)/10.)

def pycaret_mod_fit_func(inp_trval_data_Xy, inp_model_lst,  targ_var='RSRP'):
    
    rcvdsrp_models_setup = setup(data = inp_trval_data_Xy,  target = targ_var, train_size=0.8, preprocess=False, transformation=False, session_id=123);
    # trained_models_lists = compare_models(n_select=len(inp_model_lst), include = inp_model_lst, fold = 5,  round = 11,  sort = 'rmse', turbo = True, verbose = True);
    trained_models_lists = compare_models(fold = 5, n_select=len(inp_model_lst), include = inp_model_lst, round = 11,  sort = 'rmse', turbo = True, verbose = True); 
    trained_models_table = pull();
    
    return trained_models_lists, trained_models_table

def pycaret_mod_perf_testing(trained_models_lists, trained_models_table, true_test_data_XX, true_test_data_yy):
    cbm_tst_columns = [];
        
    for x in np.arange(0, len(trained_models_lists)):

        cb_mod_trn_mobj = trained_models_lists[x];                      # Model object in the table;
        cb_mod_trn_indx = trained_models_table.index.values[x];         # Model indexx in the table;
        cb_mod_trn_name = trained_models_table.Model[x];                # Model name   in the table;    
        
        pred_test_data_yy  = cb_mod_trn_mobj.predict(true_test_data_XX);

        cb_mod_tst_mabs = mean_absolute_error(            true_test_data_yy, pred_test_data_yy);
        cb_mod_tst_msee = mean_squared_error(             true_test_data_yy, pred_test_data_yy);
        cb_mod_tst_mape = mean_absolute_percentage_error( true_test_data_yy, pred_test_data_yy);
        cb_mod_tst_evar = explained_variance_score(       true_test_data_yy, pred_test_data_yy);         
        cb_mod_tst_rmse = np.sqrt(cb_mod_tst_msee);
        
        cb_mod_tst_r2sc = r2_score(true_test_data_yy, pred_test_data_yy)
        n = true_test_data_XX.shape[0]  # Number of samples in the test set
        p = true_test_data_XX.shape[1]  # Number of predictors/features

        # Calculate the adjusted R-squared
        cb_mod_tst_adr2 = 1 - ((1 - cb_mod_tst_r2sc) * (n - 1)) / (n - p - 1)

        cbm_tst_columns.append([cb_mod_trn_indx, cb_mod_trn_name, cb_mod_tst_mabs, cb_mod_tst_msee, cb_mod_tst_rmse, cb_mod_tst_mape, cb_mod_tst_evar, cb_mod_tst_r2sc, cb_mod_tst_adr2]);
        
    test_data_perfor_df = pd.DataFrame(data=cbm_tst_columns, columns=['test_abbr', 'test_name', 'test_mabs', 'test_msee', 'test_rmse', 'test_mape', 'test_expv', 'test_r2sc', 'test_adr2']);
   
    return test_data_perfor_df

# %% Data Loading and preprocessing
# %%% Data loading

SDataa_aRow_aFeat_XY = pd.read_csv('prop_mod_samp_data.csv');
col_name_sel_feat = []
col_name_sim_data = SDataa_aRow_aFeat_XY.columns.values
for xi in np.arange(0, col_name_sim_data.shape[0]):
    cur_col_name = col_name_sim_data[xi];
    cur_col_data = SDataa_aRow_aFeat_XY[cur_col_name].values;
    cur_dat_ushp = SDataa_aRow_aFeat_XY[cur_col_name].unique().shape[0];

    if cur_dat_ushp<2:
        print(f'The column: {cur_col_name} has {cur_dat_ushp} values');
    else:
        col_name_sel_feat.append(cur_col_name);
        
        
SDataa_aRow_aFeat_XY['BS_Tilt']   = SDataa_aRow_aFeat_XY['BS_Tilt'].astype('float64');
SDataa_aRow_aFeat_XY['BS_Height'] = SDataa_aRow_aFeat_XY['BS_Height'].astype('float64');
SDataa_aRow_aFeat_XY['LOS']       = SDataa_aRow_aFeat_XY['LOS'].astype('float64');   
        
# %%% Data analysis and visualization

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

# Setting the figure size and DPI
plt.figure(figsize=(20, 10), dpi=300)

# List of features to plot
features = ['BS_Tilt', 'BS_Azimuth', 'BS_Height', 'Distance', 'UE_Azimuth', 'UE_Tilt', 'LOS', 'Indoor_Path', 'Outdoor_Path', 'RSRP']

# Number of rows and columns in the subplot grid
n_rows = 2
n_cols = 5

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), dpi=300)
fig.suptitle('Distribution of Features and Target Variable (RSRP) in Propagation Modeling Data', fontsize=20)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each feature's distribution
for i, feature in enumerate(features):
    if i == 9:
        sns.histplot(SDataa_aRow_aFeat_XY[feature], bins=12, kde=True, ax=axes[i], color='r')
        axes[i].set_title(feature, fontsize=16, color='r')
    else:
        sns.histplot(SDataa_aRow_aFeat_XY[feature], bins=12, kde=True, ax=axes[i], color='g')
        axes[i].set_title( feature, fontsize=16, color='g')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)

    # Set y-axis to scientific notation
    axes[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[i].yaxis.get_offset_text().set_size(12)  # Set font size for the offset text (e.g., x1e3)
    axes[i].tick_params(axis='x', labelsize=14)
    axes[i].tick_params(axis='y', labelsize=14)
    
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
sav_fold = 'Results_saved_figs'
plt.savefig(os.path.join(sav_fold, 'feat_dist_comb.png'))
plt.savefig(os.path.join(sav_fold, 'feat_dist_comb.pdf'))

# Show the plot
plt.show()

# %% Model Training Part
# %%% Data seperation with respect to feature value

XX_Colum            = ['BS_Tilt', 'BS_Azimuth', 'BS_Height', 'Distance', 'UE_Azimuth', 'UE_Tilt', 'LOS', 'Indoor_Path', 'Outdoor_Path']
yy_Colum            = ['RSRP']

com_models_names_sl = ['catboost', 'lightgbm', 'xgboost', 'knn', 'et', 'rf', 'gbr', 'ada', 'lr',
                       'lar', 'ridge', 'huber', 'br', 'lasso', 'llar', 'dt', 'en', 'omp', 'par', 'dummy'];
com_models_names_xi = ['catboost', 'lightgbm', 'xgboost', 'knn', 'ada', 'gbr', 'et', 'rf', 'dt'];


STrval_sRow_sFeat_XY, STests_sRow_sFeat_XY = train_test_split(SDataa_aRow_aFeat_XY, random_state=42, test_size=0.2);
STrain_sRow_sFeat_XY, SValid_sRow_sFeat_XY = train_test_split(STrval_sRow_sFeat_XY, random_state=42, test_size=0.2);

# %%%


XX_Train            = STrain_sRow_sFeat_XY[XX_Colum];
yy_Train            = STrain_sRow_sFeat_XY[yy_Colum];

XX_Valid            = SValid_sRow_sFeat_XY[XX_Colum];
yy_Valid            = SValid_sRow_sFeat_XY[yy_Colum];

XX_Trval            = STrval_sRow_sFeat_XY[XX_Colum];
yy_Trval            = STrval_sRow_sFeat_XY[yy_Colum];

XX_Tests            = STests_sRow_sFeat_XY[XX_Colum];
yy_Tests            = STests_sRow_sFeat_XY[yy_Colum];

ss_scal_train       = StandardScaler();mm_scal_train = MinMaxScaler();
ss_scal_valid       = StandardScaler();mm_scal_valid = MinMaxScaler();
ss_scal_trval       = StandardScaler();mm_scal_trval = MinMaxScaler();
ss_scal_tests       = StandardScaler();mm_scal_tests = MinMaxScaler();


ss_scal_train.fit(XX_Train);
XX_Train_Scld       = ss_scal_train.transform(XX_Train);

ss_scal_valid.fit(XX_Valid);
XX_Valid_Scld       = ss_scal_valid.transform(XX_Valid);

ss_scal_trval.fit(XX_Trval);
XX_Trval_Scld       = ss_scal_trval.transform(XX_Trval);

ss_scal_trval.fit(XX_Tests);
XX_Tests_Scld       = ss_scal_trval.transform(XX_Tests);

# %%% Model Training

# Provide your own X_train and y_train data
XX_Train_tensr      = torch.from_numpy(XX_Train_Scld).float();
yy_Train_tensr      = torch.from_numpy(yy_Train.values).float();

# Provide your own X_valid and y_valid data
XX_Valid_tensr      = torch.from_numpy(XX_Valid_Scld).float();
yy_Valid_tensr      = torch.from_numpy(yy_Valid.values).float();

# Provide your own X_trval and y_trval data
XX_Trval_tensr      = torch.from_numpy(XX_Trval_Scld).float();
yy_Trval_tensr      = torch.from_numpy(yy_Trval.values).float();


XY_col_names        = XX_Colum + yy_Colum

XY_Train_scld_np    = np.concatenate((XX_Train_Scld, yy_Train.values),  axis=1);
XY_Train_scld_df    = pd.DataFrame(data=XY_Train_scld_np, columns=XY_col_names);

XY_Valid_scld_np    = np.concatenate((XX_Valid_Scld, yy_Valid.values),  axis=1);
XY_Valid_scld_df    = pd.DataFrame(data=XY_Valid_scld_np, columns=XY_col_names);

XY_Trval_scld_np    = np.concatenate((XX_Trval_Scld, yy_Trval.values),  axis=1);
XY_Trval_scld_df    = pd.DataFrame(data=XY_Trval_scld_np, columns=XY_col_names);

XY_Tests_scld_np    = np.concatenate((XX_Tests_Scld, yy_Tests.values),  axis=1);
XY_Tests_scld_df    = pd.DataFrame(data=XY_Tests_scld_np, columns=XY_col_names);


trn_models_lists_xi, trn_models_table_xi = pycaret_mod_fit_func(XY_Trval_scld_df, com_models_names_xi);

# %%% Model Testing

perf_evals_models   = pycaret_mod_perf_testing(trn_models_lists_xi, trn_models_table_xi, XX_Tests_Scld, yy_Tests);


# %% Shap Analysis

import shap 
final_model = trn_models_lists_xi[0]

# %%% Shap bar plot

# Initialize the SHAP explainer
explainer = shap.Explainer(final_model)
shap_values = explainer(XY_Trval_scld_df.drop(columns=['RSRP']))


# Set up the figure with desired DPI
plt.figure(dpi=300)  # Change the DPI value as needed
fig_size = (10, 6)
# Create the SHAP summary plot with custom parameters
ax = shap.summary_plot(
    shap_values, 
    XX_Trval,
    plot_size=fig_size,  # Width and height in inches
    max_display=10,     # Maximum number of features to display
    plot_type="bar",    # 'dot',1 'bar', or 'violin'
    show=False,          # Don't show the plot immediately
)

# Customize the plot using the axis object
plt.title( 'Shap summary plot for Propagation modeling data', fontsize=16)
plt.xlabel('SHAP Value (Impact on Model Output)',             fontsize=14)
plt.ylabel('Propagation modeling features',                   fontsize=14)

# Adding text annotations to the right side of the bars
ax = plt.gca()
for rect in ax.patches:
    width = rect.get_width()
    ax.text(
        width + 0.1,  # X-coordinate for text (slightly to the right of the bar)
        rect.get_y() + rect.get_height() / 2,  # Y-coordinate for text (centered vertically)
        f'{width:.2f}',  # Text to display (formatted to 2 decimal places)
        ha='left',  # Horizontal alignment
        va='center',  # Vertical alignment
        fontsize=12  # Font size
    )


plt.tight_layout()
sav_fold = 'Results_saved_figs'
plt.savefig(os.path.join(sav_fold, 'shap_summary_plot_bars.png'))
plt.savefig(os.path.join(sav_fold, 'shap_summary_plot_bars.pdf'))

# Show the plot
plt.show()

# %%% Shap summary plot

# Initialize the SHAP explainer
explainer = shap.Explainer(final_model)
shap_values = explainer(XY_Trval_scld_df.drop(columns=['RSRP']))


# Set up the figure with desired DPI
plt.figure(dpi=300)  # Change the DPI value as needed
fig_size = (10, 6)
# Create the SHAP summary plot with custom parameters
ax = shap.summary_plot(
    shap_values, 
    XX_Trval,
    plot_size=fig_size,  # Width and height in inches
    max_display=10,     # Maximum number of features to display
    plot_type="dot",    # 'dot',1 'bar', or 'violin'
    show=False          # Don't show the plot immediately
)


# Customize the plot using the axis object
plt.title( 'Shap summary plot for Propagation modeling data', fontsize=16)
plt.xlabel('SHAP Value (Impact on Model Output)',             fontsize=14)
plt.ylabel('Propagation modeling features',                   fontsize=14)

plt.tight_layout()
# sav_fold = 'MF_Results_Fst_Phase/Shap Mod Fitting Results'
plt.savefig(os.path.join(sav_fold, 'shap_summary_plot.png'))
plt.savefig(os.path.join(sav_fold, 'shap_summary_plot.pdf'))

# Show the plot
plt.show()

# %%% Shap dependence plot

# Get feature names
feature_names = XY_Trval_scld_df.drop(columns=['RSRP']).columns
# Initialize the SHAP explainer and get shap values as numpy array
shap_values = explainer.shap_values(XY_Trval_scld_df.drop(columns=['RSRP']))

# Loop over each feature and create a SHAP dependence plot
for feature in feature_names:
   
    # Create the SHAP dependence plot
    shap.dependence_plot(
        feature, 
        shap_values, 
        XX_Trval,
        show=False  # Don't show the plot immediately
    )
    
    # Customize the plot
    plt.gcf().set_dpi(300)  # Set the DPI for high-quality figures
    plt.gcf().set_size_inches(fig_size)  # Set the figure size
    plt.title(f'SHAP Dependence Plot for {feature}', fontsize=16)
    plt.xlabel(f'{feature}', fontsize=14)
    plt.ylabel('SHAP Value', fontsize=14)
    
    plt.tight_layout()
    # sav_fold = 'MF_Results_Fst_Phase/Shap Mod Fitting Results'
    plt.savefig(os.path.join(sav_fold, f'shap_depend_plot_Feat_{feature}.png'))
    plt.savefig(os.path.join(sav_fold, f'shap_depend_plot_Feat_{feature}.pdf'))

    # Show the customized plot
    plt.show()

# %%% Shap Interaction plots

import shap
import matplotlib.pyplot as plt
import itertools

# Assuming 'explainer' is already defined and XY_Trval_scld_df contains your data

# Initialize the SHAP explainer and get shap values as a numpy array
shap_values = explainer.shap_values(XY_Trval_scld_df.drop(columns=['RSRP']))

# Get feature names
feature_names = XY_Trval_scld_df.drop(columns=['RSRP']).columns

# Loop over each pair of features to create SHAP interaction plots
for feature1, feature2 in itertools.combinations(feature_names, 2):
    # plt.figure(dpi=300)
    
    # # Create the SHAP interaction plot
    # shap.interaction_plot(
    #     shap_values[:, feature1], 
    #     shap_values[:, feature2], 
    #     XY_Trval_scld_df.drop(columns=['RSRP']),
    #     show=False  # Don't show the plot immediately
    # )

    # Create the SHAP interaction plot
    shap.dependence_plot(
        feature1, 
        shap_values, 
        XX_Trval,
        interaction_index=feature2,
        show=False  # Don't show the plot immediately
    )
    
    # Customize the plot
    plt.gcf().set_dpi(300)  # Set the DPI for high-quality figures
    plt.gcf().set_size_inches(fig_size)  # Set the figure size
    plt.title(f'SHAP Interaction Plot for {feature1} and {feature2}', fontsize=16)
    plt.xlabel(f'{feature1}', fontsize=14)
    plt.ylabel(f'{feature2}', fontsize=14)
    
    
    plt.tight_layout()
    # sav_fold = 'MF_Results_Fst_Phase/Shap Mod Fitting Results'
    plt.savefig(os.path.join(sav_fold, f'shap_interact_plot_F1_{feature1}__F2_{feature2}.png'))
    plt.savefig(os.path.join(sav_fold, f'shap_interact_plot_F1_{feature1}__F2_{feature2}.pdf'))
    
    plt.show()

# %%% Shap force 

import shap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define the list of sample indices
sample_indices = [0, 1, 9, 10, 11, 99, 245, 250, 455, 999, 4554]  # Replace with your desired indices
# sample_indices = [9, 4554]  # Replace with your desired indices
fig_size = (12, 6)  # Example figure size

# Loop over each instance in the dataset to create SHAP force and waterfall plots
for i in sample_indices:
    
    # Get the original feature names and rounded feature values
    feature_names = XY_Trval_scld_df.drop(columns=['RSRP']).columns.tolist()
    rounded_features = np.round(XX_Trval.iloc[i].values, 2)
    
    print(np.round(XX_Trval.reset_index().iloc[i], 2))
    
    # Create custom labels with feature name on the first line and value on the second line
    custom_labels = [f"\n\n\n{feature}\n" for feature, value in zip(feature_names, rounded_features)]

    
    # Create the SHAP force plot for a single instance
    shap.force_plot(
        explainer.expected_value, 
        np.round(shap_values[i], 2), 
        features=rounded_features,
        feature_names=custom_labels,
        matplotlib=True,  # Render the plot with Matplotlib
        show=False        # Don't show the plot immediately
    )
    
    # Customize the plot
    plt.gcf().set_dpi(300)  # Set the DPI for high-quality figures
    plt.gcf().set_size_inches((12, 6))  # Set the figure size

    # Increase font size of text on bars
    for txt in plt.gcf().findobj(match=plt.Text):
        txt.set_fontsize(15)  # Adjust the value as needed
    
    # Customize the force plot
    plt.title(f'SHAP Force Plot for Instance {i}', fontsize=16, y=1.75)
    plt.xlabel('SHAP Value', fontsize=16)  # Set x-axis label and font size
    plt.ylabel('Features',   fontsize=16)  # Set y-axis label and font size
    plt.xticks(fontsize=14)  # Set x-tick font size
    plt.yticks(fontsize=14)  # Set y-tick font size
    
    
    
    plt.tight_layout()
    # sav_fold = 'MF_Results_Fst_Phase/Shap Mod Fitting Results'
    plt.savefig(os.path.join(sav_fold, f'shap_forceabcd_plot__Ind_{i}.png'));
    plt.savefig(os.path.join(sav_fold, f'shap_forceabcd_plot__Ind_{i}.pdf'));

    # Show the customized force plot
    plt.show()
    
    ## %%% Shap waterfall     
    shap_explanation = shap.Explanation(
        values=np.round(shap_values[i], 2),
        base_values=np.round(explainer.expected_value, 2),
        data=rounded_features,
        feature_names=XY_Trval_scld_df.drop(columns=['RSRP']).columns.tolist()
    )
    
    
    
    # Create the SHAP waterfall plot for a single instance
    fig_wf = shap.waterfall_plot(
        shap_explanation,
        show=False  # Don't show the plot immediately
    )
    
    # Customize the plot
    plt.gcf().set_dpi(300)  # Set the DPI for high-quality figures
    plt.gcf().set_size_inches((11, 6))  # Set the figure size
    plt.title(f'SHAP Waterfall Plot for Instance {i}', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=16)  # Set x-axis label and font size
    plt.xticks(fontsize=14)  # Set x-tick font size
    
    # Increase font size of text on bars
    # for txt in fig_wf.axes[0].get_yticklabels():
    #     txt.set_fontsize(24) # Adjust the value as needed
    
    # Increase font size of text on bars
    for txt in plt.gcf().findobj(match=plt.Text):
        txt.set_fontsize(15)  # Adjust the value as needed

    # # Increase font size of text on bars inside the waterfall plot
    # for txt in plt.gcf().findobj(match=plt.Text):
    #     if txt.get_text().strip() in feature_names:
    #         txt.set_fontsize(20)  # Adjust the value as needed for bar text    
    
    # # Increase font size of text on bars
    # for txt in fig.axes[0].texts:
    #     if txt.get_text() in feature_names:
    #         txt.set_fontsize(24) # Adjust the value as needed    
    
    # Function to format ticks to 2 decimal places
    def format_func(value, tick_number):
        return f'{value:.2f}'
    
    # Set xticks to round 2 decimal places
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func), fontsize=16)
    
    plt.tight_layout()
    # sav_fold = 'MF_Results_Fst_Phase/Shap Mod Fitting Results'
    plt.savefig(os.path.join(sav_fold, f'shap_waterfall_plot__Ind_{i}.png'));
    plt.savefig(os.path.join(sav_fold, f'shap_waterfall_plot__Ind_{i}.pdf'));
    
    plt.show()



