import pandas as pd
import matplotlib.pyplot as plt
import sys

folder = sys.argv[1]
csv = folder + '/consumption.csv'
df = pd.read_csv(csv)
df['time'] = df.index*5
df['server_cpu']/=1000000000
df['client_cpu']/=1000000000
df['server_memory'] /= 1000
df['client_memory'] /= 1000
print(df)
plt.plot(df['time'], df[['server_cpu', 'client_cpu']])
plt.legend(['Server cpu usage(#cores)', 'Client cpu usage(#cores)'])
plt.xlabel('Time(s)')
plt.title('Server and Client CPU Usage')
plt.savefig(folder + '/cpu_consumption.png')
plt.clf()
plt.plot(df['time'], df[['server_memory', 'client_memory']])
plt.legend(['Server memory usage(MB)', 'Client memory usage(MB)'])
plt.xlabel('Time(s)')
plt.title('Server and Client Memory Usage')
plt.savefig(folder + '/memory_consumption.png')