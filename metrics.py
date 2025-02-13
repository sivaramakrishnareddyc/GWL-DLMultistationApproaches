import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import itertools

# Define the classes
classes = ['annual', 'mixed', 'inertial']

# Define the datasets
datasets = ['single', 'ohe', 'stat', 'stat_ohe', 'clus_ohe', 'clus_stat','clus_stat_ohe' , 'NO', 'clus_NO'] #~'single', 'ohe', 'stat', 'stat_ohe', 'clus_ohe', 'clus_stat','clus_stat_ohe' , 'NO', 'clus_NO'

# Define the models
models = ['BILSTM', 'LSTM', 'GRU']

metrics = ['NSE', 'R2', 'MAE', 'KGE']

# Define the color and marker combinations
# color_marker_dict = {
#     'single': {'color': 'black', 'linestyle':'-'},
#     'ohe': {'color': 'blue', 'linestyle':'-'},
#     'stat': {'color': 'red', 'linestyle':'-'},
#     'stat_ohe': {'color': 'green', 'linestyle':'-'},
#     'NO': {'color': 'orange', 'linestyle': '-'},
#     'clus_ohe': {'color': 'blue', 'linestyle': 'dashed'},
#     'clus_stat': {'color': 'red', 'linestyle': 'dashed'},
#     'clus_stat_ohe': {'color': 'green', 'linestyle': 'dashed'},
#     'clus_NO': {'color': 'orange', 'linestyle': 'dashed'}
# }

# Define the color-blind friendly color and marker combinations
color_marker_dict = {
    'single': {'color': '#000000', 'linestyle': '-'},
    'ohe': {'color': '#377eb8', 'linestyle': '-'},
    'stat': {'color': '#e41a1c', 'linestyle': '-'},
    'stat_ohe': {'color': '#4daf4a', 'linestyle': '-'},
    'NO': {'color': '#ff7f00', 'linestyle': '-'},
    'clus_ohe': {'color': '#377eb8', 'linestyle': 'dashed'},
    'clus_stat': {'color': '#e41a1c', 'linestyle': 'dashed'},
    'clus_stat_ohe': {'color': '#4daf4a', 'linestyle': 'dashed'},
    'clus_NO': {'color': '#ff7f00', 'linestyle': 'dashed'}
}
for metric in metrics:
    for model in models:
        fig, axs = plt.subplots(2, len(classes), figsize=(12, 8), sharex=True, sharey='row')
        fig.suptitle(f'{model} - {metric} Comparison', fontsize=16)
        
        for j, Class in enumerate(classes):
            for i, dataset in enumerate(datasets):
                # Read the data for the specific dataset, class, model, and metric
                filename_SA = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Single_station/Standalone/scores_era5_2{model}.csv"
                filename_Wav = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Single_station/Wavelet/scores_era5{model}la8.csv"
                filename_SA_no = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/NO/Standalone/scores_era5{model}.csv"
                filename_Wav_no = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/NO/Wavelet/scores_era5{model}la8.csv"
                filename_SA_clus_NO = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_NO/Standalone/scores_era5{model}{Class}.csv"
                filename_Wav_clus_NO = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_NO/Wavelet/scores_era5{model}la8{Class}.csv"
                filename_SA_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/OHE/Standalone/scores_era5{model}.csv"
                filename_Wav_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/OHE/Wavelet/scores_era5{model}la8allFRANCE.csv"
                filename_SA_sta = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Static_variables/Standalone/scores_era5{model}.csv"
                filename_Wav_sta = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Static_variables/Wavelet/scores_era5{model}la8allFRANCE.csv"
                filename_SA_clus_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_OHE/Standalone/scores_era5{model}{Class}.csv"
                filename_Wav_clus_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_OHE/Wavelet/scores_era5{model}la8{Class}.csv"
                filename_SA_clus_stat = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_Static/Standalone/scores_era5{model}{Class}.csv"
                filename_Wav_clus_stat = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_Static/Wavelet/scores_era5{model}la8{Class}.csv"
                filename_SA_stat_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Static_OHE/Standalone/scores_era5{model}.csv"
                filename_Wav_stat_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Static_OHE/Wavelet/scores_era5{model}la8allFRANCE.csv"
                filename_SA_clus_stat_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_Static_OHE/Standalone/scores_era5{model}{Class}.csv"
                filename_Wav_clus_stat_ohe = f"/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Cluster_Static_OHE/Wavelet/scores_era5{model}la8{Class}.csv"
                if dataset == 'NO':
                    df_SA = pd.read_csv(filename_SA_no)
                    df_Wav = pd.read_csv(filename_Wav_no)
                elif dataset == 'clus_NO':
                    df_SA = pd.read_csv(filename_SA_clus_NO)
                    df_Wav = pd.read_csv(filename_Wav_clus_NO)
                elif dataset == 'ohe':
                    df_SA = pd.read_csv(filename_SA_ohe)
                    df_Wav = pd.read_csv(filename_Wav_ohe)
                elif dataset == 'stat':
                    df_SA = pd.read_csv(filename_SA_sta)
                    df_Wav = pd.read_csv(filename_Wav_sta)
                elif dataset == 'clus_ohe':
                    df_SA = pd.read_csv(filename_SA_clus_ohe)
                    df_Wav = pd.read_csv(filename_Wav_clus_ohe)
                elif dataset == 'clus_stat':
                    df_SA = pd.read_csv(filename_SA_clus_stat)
                    df_Wav = pd.read_csv(filename_Wav_clus_stat)
                elif dataset == 'stat_ohe':
                    df_SA = pd.read_csv(filename_SA_stat_ohe)
                    df_Wav = pd.read_csv(filename_Wav_stat_ohe)
                elif dataset == 'clus_stat_ohe':
                    df_SA = pd.read_csv(filename_SA_clus_stat_ohe)
                    df_Wav = pd.read_csv(filename_Wav_clus_stat_ohe)
                else:
                    df_SA = pd.read_csv(filename_SA)
                    df_Wav = pd.read_csv(filename_Wav)

                # Filter the data for the specific class and metric
                metric_values_SA = df_SA[(df_SA['Class'] == Class)][f'{metric}_te']
                metric_values_Wav = df_Wav[(df_Wav['Class'] == Class)][f'{metric}_te']

                # Calculate CDF for SA
                cdf_SA = stats.cumfreq(metric_values_SA, numbins=metric_values_SA.shape[0])
                x_SA = cdf_SA.lowerlimit + np.linspace(0, cdf_SA.binsize * cdf_SA.cumcount.size, cdf_SA.cumcount.size)
                y_SA = cdf_SA.cumcount / len(metric_values_SA)

                # Calculate CDF for Wav
                cdf_Wav = stats.cumfreq(metric_values_Wav, numbins=metric_values_Wav.shape[0])
                x_Wav = cdf_Wav.lowerlimit + np.linspace(0, cdf_Wav.binsize * cdf_Wav.cumcount.size, cdf_Wav.cumcount.size)
                y_Wav = cdf_Wav.cumcount / len(metric_values_Wav)

                # Plot SA
                
                axs[0, j].plot(x_SA, y_SA, label=f'{dataset} - SA',color=color_marker_dict[dataset]['color'], linestyle=color_marker_dict[dataset]['linestyle'])

                # Plot Wav
                axs[1, j].plot(x_Wav, y_Wav, label=f'{dataset} - Wav',color=color_marker_dict[dataset]['color'], linestyle=color_marker_dict[dataset]['linestyle'])

            axs[0, j].set_title(f'{Class}', fontsize=14)
            axs[0, 0].set_ylabel('CDF (Standalone)', fontsize=14)
            axs[1, 0].set_ylabel('CDF (Wavelet)', fontsize=14)
            axs[1, j].set_xlim(-1, 1)  # Set x-axis limits to -1 and 1
            if j == 0:  # Add legend only to the first column of subplots in each row
                axs[0, j].legend(fontsize=12)
                axs[1, j].legend(fontsize=12)
            axs[0, j].tick_params(axis='both', which='major', labelsize=14)
            axs[1, j].tick_params(axis='both', which='major', labelsize=14)
            axs[0, j].grid(True)
            axs[1, j].grid(True)

        # axs[0, -1].set_xlabel(f'{metric} Value', fontsize=12)
        axs[1, 1].set_xlabel(f'{metric} Value', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{model}_{metric}_comparison.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()