'''
This file is for plotting the log metrics.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    csv_folder = 'logs/experiment4/'
    cloud_df = pd.read_csv(csv_folder+'cloud.csv', header=None)
    cloud_df.rename(columns={0:'received', 1:'updated', 2:'saved', 3:'encoded', 4:'sent'}, inplace=True)
    cloud_start_time = cloud_df['received'].iloc[0]
    cloud_df -= cloud_start_time
    edge_inf_df = pd.read_csv(csv_folder+'edge_inf.csv', header=None)
    edge_inf_df.rename(columns={0:'end_time', 1:'avg_inf_time', 2:'avg_wait_time', 3:'avg_queue_len'}, inplace=True)
    edge_update_df = pd.read_csv(csv_folder+'edge_update.csv', header=None)
    edge_update_df.rename(columns={0:'update_start', 1:'prepare', 2:'send_start', 3:'send_end', 4:'receive', 5:'eval_start', 6:'eval_end'}, inplace=True)
    edge_start_time = edge_inf_df['end_time'].iloc[0]
    edge_inf_df['end_time'] -= edge_start_time
    edge_inf_df['avg_queue_len'] = np.log(edge_inf_df['avg_queue_len']+0.001)/np.log(2)
    edge_update_df -= edge_start_time
    print(cloud_df.head())
    print(edge_inf_df.head())
    print(edge_update_df.head())
    edge_inf_df = edge_inf_df[:50]
    edge_update_df = edge_update_df[:50]
    plt.plot(edge_inf_df['end_time'], edge_inf_df[['avg_inf_time', 'avg_wait_time', 'avg_queue_len']])
    plt.legend(['Average inference time', 'Average waiting time', 'Log of Average queue length'])
    plt.savefig(csv_folder+'edge_info.png')
    for index, row in edge_update_df.iterrows():
        if row['eval_end'] >= edge_inf_df['end_time'].iloc[len(edge_inf_df)-1]:
            break
        plt.axvspan(row['update_start'], row['eval_end'], color='grey', alpha=0.5)
    plt.savefig(csv_folder+'edge_info_with_update.png')
    for index, row in edge_update_df.iterrows():
        if row['eval_end'] >= edge_inf_df['end_time'].iloc[len(edge_inf_df)-1]:
            break
        plt.axvspan(row['update_start'], row['send_start'], color='red', alpha=0.5)
        plt.axvspan(row['send_start'], row['send_end'], color='yellow', alpha=0.5)
        plt.axvspan(row['receive'], row['eval_start'], color='green', alpha=0.5)
        plt.axvspan(row['eval_start'], row['eval_end'], color='blue', alpha=0.5)
    plt.savefig(csv_folder+'edge_info_with_detailed_update.png')
    return


if __name__ == '__main__':
    plot()