'''
This file is for plotting the log metrics.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
This file is for plotting the log metrics.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plotdir = "multiplots/"
num_edge = 2
def plot_edge():
    csv_folders = ['logs/multiexp1/', 'logs/multiexp2/', 'logs/multiexp3/','logs/multiexp4/']
    # csv_folders = ['logs/multiexp1/', 'logs/multiexp2/', 'logs/multiexp3/', 'logs/multiexp4/', 'logs/multiexp5/']
    edge_inf_dfs = []
    edge_update_dfs = []
    configs = []
    progresses = []
    for csv_folder in csv_folders:
        with open(csv_folder + 'comment.txt') as f:
            text = f.readline()[:-1]
        for eid in range(0, num_edge):
            edge_exp_folder = csv_folder + str(eid) + '/'
            edge_inf_df = pd.read_csv(edge_exp_folder+'edge_inf.csv', header=None)
            edge_inf_df.rename(columns={0:'end_time', 1:'avg_inf_time', 2:'avg_wait_time', 3:'avg_queue_len'}, inplace=True)
            edge_update_df = pd.read_csv(edge_exp_folder+'edge_update.csv', header=None)
            edge_update_df.rename(columns={0:'update_start', 1:'prepare', 2:'send_start', 3:'send_end', 4:'receive', 5:'update_end'}, inplace=True)
            edge_start_time = edge_inf_df['end_time'].iloc[0]
            edge_inf_df['end_time'] -= edge_start_time
            edge_inf_df['processed images'] = edge_inf_df.index * 10
            edge_update_df -= edge_start_time
            edge_inf_df = edge_inf_df[edge_inf_df['end_time'] <= 400]
            edge_inf_dfs.append(edge_inf_df)
            edge_update_dfs.append(edge_update_df)
            progresses.append(edge_inf_df['processed images'].iloc[-1]/edge_inf_df['end_time'].iloc[-1])
            configs.append(text +' :' +str(eid))
    for edge_inf_df in edge_inf_dfs:
        plt.plot(edge_inf_df['end_time'], edge_inf_df[['avg_inf_time']], marker='.')
        
    plt.legend(configs)
    plt.xlabel('Time(s)')
    plt.ylabel('Time(s)')
    plt.title('Average Inference Time under Different Settings')
    plt.savefig(plotdir+'avg_inf.png')
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
    plt.ylabel('Time(s)')
    ax.set_title('Edge Average Update Time')
    plt.xticks(rotation=35)
    plt.subplots_adjust(bottom=0.2)
    ax.figure.savefig(plotdir + 'edge_update.png')
    plt.clf()

if __name__ == '__main__':

    plot_edge()
