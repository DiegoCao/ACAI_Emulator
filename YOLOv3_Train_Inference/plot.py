'''
This file is for plotting the log metrics.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    csv_folders = ['logs/experiment13/', 'logs/experiment15/']
    # csv_folders = ['logs/experiment13/', 'logs/experiment11/', 'logs/experiment14/', 'logs/experiment12/' ]
    cloud_dfs = []
    edge_inf_dfs = []
    edge_update_dfs = []
    configs = []
    for csv_folder in csv_folders:
        with open(csv_folder + 'comment.txt') as f:
            text = f.readline()[:-1]
            configs.append(text)
        cloud_df = pd.read_csv(csv_folder+'cloud.csv', header=None)
        cloud_df.rename(columns={0:'received', 1:'updated', 2:'saved', 3:'encoded', 4:'sent'}, inplace=True)
        cloud_start_time = cloud_df['received'].iloc[0]
        cloud_df -= cloud_start_time
        edge_inf_df = pd.read_csv(csv_folder+'edge_inf.csv', header=None)
        edge_inf_df.rename(columns={0:'end_time', 1:'avg_inf_time', 2:'avg_wait_time', 3:'avg_queue_len'}, inplace=True)
        edge_update_df = pd.read_csv(csv_folder+'edge_update.csv', header=None)
        edge_update_df.rename(columns={0:'update_start', 1:'prepare', 2:'send_start', 3:'send_end', 4:'receive', 5:'update_end'}, inplace=True)
        edge_start_time = edge_inf_df['end_time'].iloc[0]
        edge_inf_df['end_time'] -= edge_start_time
        edge_inf_df['processed images'] = edge_inf_df.index * 10
        edge_update_df -= edge_start_time
        edge_inf_df = edge_inf_df[edge_inf_df['end_time'] <= 400]
        cloud_dfs.append(cloud_df)
        edge_inf_dfs.append(edge_inf_df)
        edge_update_dfs.append(edge_update_df)
    for edge_inf_df in edge_inf_dfs:
        plt.plot(edge_inf_df['end_time'], edge_inf_df[['avg_inf_time']], marker='.')
    plt.legend(configs)
    plt.xlabel('Time(s)')
    plt.title('Average Inference Time under Different Settings(s)')
    plt.savefig('plots/avg_inf.png')
    plt.clf()
    for edge_inf_df in edge_inf_dfs:
        plt.plot(edge_inf_df['end_time'], edge_inf_df[['avg_wait_time']], marker='.')
    plt.legend(configs)
    plt.xlabel('Time(s)')
    plt.title('Average Waiting Time under Different Settings(s)')
    plt.savefig('plots/avg_wait.png')
    plt.clf()
    for edge_inf_df in edge_inf_dfs:
        plt.plot(edge_inf_df['end_time'], edge_inf_df[['avg_queue_len']], marker='.')
    plt.legend(configs)
    plt.xlabel('Time(s)')
    plt.title('Average Queue Length under Different Settings(s)')
    plt.savefig('plots/avg_queue.png')
    plt.clf()
    for edge_inf_df in edge_inf_dfs:
        plt.plot(edge_inf_df['end_time'], edge_inf_df[['processed images']], marker='.')
    plt.legend(configs)
    plt.xlabel('Time(s)')
    plt.title('Processed Images under Different Settings(s)')
    plt.savefig('plots/processed.png')
    plt.clf()
    edge_update_df_means = []
    for edge_update_df in edge_update_dfs:
        edge_update_df['prepare_time'] = edge_update_df['send_start'] - edge_update_df['update_start']
        edge_update_df['send_time'] = edge_update_df['send_end'] - edge_update_df['send_start']
        edge_update_df['receive_time'] = edge_update_df['receive'] - edge_update_df['send_end']
        edge_update_df['load_time'] = edge_update_df['update_end'] - edge_update_df['receive']
        edge_update_df_mean = edge_update_df[['prepare_time', 'send_time', 'receive_time', 'load_time']].mean()
        edge_update_df_means.append(edge_update_df_mean)
    edge_update_time = pd.DataFrame(edge_update_df_means, columns = ['prepare_time', 'send_time', 'receive_time', 'load_time'])
    edge_update_time.columns = ['Prepare Images', 'Send Images', 'Receive Updates', 'Load Model']
    edge_update_time.index = configs
    print(edge_update_time)
    ax = edge_update_time.plot.bar(stacked = True)
    ax.set_title('Edge Average Update Time(s)')
    plt.xticks(rotation=0)
    # plt.subplots_adjust(bottom=0.3)
    ax.figure.savefig('plots/edge_update.png')
    plt.clf()
    cloud_df_means = []
    for cloud_df in cloud_dfs:
        cloud_df['Model Refine Time'] = cloud_df['updated'] - cloud_df['received']
        cloud_df['New Model Save Time'] = cloud_df['saved'] - cloud_df['updated']
        cloud_df['Prepare Send Img Time'] = cloud_df['encoded'] - cloud_df['saved']
        cloud_df['Msg Sent Time'] = cloud_df['sent'] - cloud_df['encoded']
        cloud_df_mean = cloud_df[['Model Refine Time', 'New Model Save Time', 'Prepare Send Img Time', 'Msg Sent Time']].mean()
        cloud_df_means.append(cloud_df_mean)
    cloud_update_time = pd.DataFrame(cloud_df_means, columns=['Model Refine Time', 'New Model Save Time', 'Prepare Send Img Time', 'Msg Sent Time'])
    cloud_update_time.index = configs
    print(cloud_update_time)
    ax = cloud_update_time.plot.bar(stacked = True)
    ax.set_title('Cloud Average Update Time(s)')
    plt.xticks(rotation=0)
    ax.figure.savefig('plots/cloud_update.png')
    plt.clf()
    ax = cloud_update_time[['New Model Save Time', 'Prepare Send Img Time', 'Msg Sent Time']].plot.bar(stacked = True)
    ax.set_title('Cloud Average Update Time Without Model Refine(s)')
    plt.xticks(rotation=0)
    ax.figure.savefig('plots/cloud_update_no_refine.png')
    return


if __name__ == '__main__':
    plot()