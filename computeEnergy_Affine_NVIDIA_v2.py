import sys
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


DEBUG = 0
PLOT = 1


#plt.rcParams.update({'font.size': 22})

reportCL = sys.argv[1]
reportSmi = sys.argv[2]

print("Input,%s,%s" % (reportCL, reportSmi))

start_write = ''
finish_write = ''

start_EXEC_FULL_2CP = ''
finish_EXEC_FULL_2CP = ''
start_READ_FULL_2CP = ''
finish_READ_FULL_2CP = ''

start_EXEC_FULL_3CP = ''
finish_EXEC_FULL_3CP = ''
start_READ_FULL_3CP = ''
finish_READ_FULL_3CP = ''

start_EXEC_HALF_2CP = ''
finish_EXEC_HALF_2CP = ''
start_READ_HALF_2CP = ''
finish_READ_HALF_2CP = ''

start_EXEC_HALF_3CP = ''
finish_EXEC_HALF_3CP = ''
start_READ_HALF_3CP = ''
finish_READ_HALF_3CP = ''

start_host = ''

start_write_l = []
finish_write_l = []

start_EXEC_FULL_2CP_l = []
finish_EXEC_FULL_2CP_l = []
start_READ_FULL_2CP_l = []
finish_READ_FULL_2CP_l = []
start_EXEC_FULL_3CP_l = []
finish_EXEC_FULL_3CP_l = []
start_READ_FULL_3CP_l = []
finish_READ_FULL_3CP_l = []
start_EXEC_HALF_2CP_l = []
finish_EXEC_HALF_2CP_l = []
start_READ_HALF_2CP_l = []
finish_READ_HALF_2CP_l = []
start_EXEC_HALF_3CP_l = []
finish_EXEC_HALF_3CP_l = []
start_READ_HALF_3CP_l = []
finish_READ_HALF_3CP_l = []

# These include the execution and read of each CP, combined, for simplicity
start_FULL_2CP_l = []
interface_FULL_2CP_FULL_3CP_l = []
interface_FULL_3CP_HALF_2CP_l = []
interface_HALF_2CP_HALF_3CP_l = []
finish_HALF_3CP_l = []

delta_start_FULL_2CP_l = []
delta_interface_FULL_2CP_FULL_3CP_l = []
delta_interface_FULL_3CP_HALF_2CP_l = []
delta_interface_HALF_2CP_HALF_3CP_l = []
delta_finish_HALF_3CP_l = []
delta_start_write_l = []
delta_finish_write_l = []

## START TRACING THE START AND END TIMES OF EACH STAGE FROM THE CL REPORT
f = open(reportCL)
format_string = "%H:%M:%S.%f"
for line in f:
	if('START HOST ' in line):
		start_host = line.split('@')[1].strip(' \n\t')
		start_host = datetime.strptime(start_host, format_string)

	if('START WRITE SAMPLES MEMOBJ' in line):
		start_write = line.split('@')[1].strip(' \n\t')
		start_write = datetime.strptime(start_write, format_string)
		start_write_l.append(start_write)

	if('FINISH WRITE SAMPLES MEMOBJ' in line):
		finish_write = line.split('@')[1].strip(' \n\t')
		finish_write = datetime.strptime(finish_write, format_string)
		finish_write_l.append(finish_write)

	if('START EXEC FULL 2 CPs' in line):
		start_EXEC_FULL_2CP = line.split('@')[1].strip(' \n\t')
		start_EXEC_FULL_2CP = datetime.strptime(start_EXEC_FULL_2CP, format_string)
		start_FULL_2CP_l.append(start_EXEC_FULL_2CP)

	if('START EXEC FULL 3 CPs' in line):
		start_EXEC_FULL_3CP = line.split('@')[1].strip(' \n\t')
		start_EXEC_FULL_3CP = datetime.strptime(start_EXEC_FULL_3CP, format_string)
		interface_FULL_2CP_FULL_3CP_l.append(start_EXEC_FULL_3CP)

	if('START EXEC HALF 2 CPs' in line):
		start_EXEC_HALF_2CP = line.split('@')[1].strip(' \n\t')
		start_EXEC_HALF_2CP = datetime.strptime(start_EXEC_HALF_2CP, format_string)
		interface_FULL_3CP_HALF_2CP_l.append(start_EXEC_HALF_2CP)

	if('FINISH READ HALF 3 CPs' in line):
		finish_READ_HALF_3CP = line.split('@')[1].strip(' \n\t')
		finish_READ_HALF_3CP = datetime.strptime(finish_READ_HALF_3CP, format_string)
		finish_HALF_3CP_l.append(finish_READ_HALF_3CP)

f.close()


# Compute the markers 


for st in start_write_l:
	delta_start_write_l.append((st - start_host).total_seconds()*1000)

for st in finish_write_l:
	delta_finish_write_l.append((st - start_host).total_seconds()*1000)

for st in start_FULL_2CP_l:
	delta_start_FULL_2CP_l.append((st - start_host).total_seconds()*1000)

for st in interface_FULL_2CP_FULL_3CP_l:
	delta_interface_FULL_2CP_FULL_3CP_l.append((st - start_host).total_seconds()*1000)

for st in interface_FULL_3CP_HALF_2CP_l:
	delta_interface_FULL_3CP_HALF_2CP_l.append((st - start_host).total_seconds()*1000)

for st in finish_HALF_3CP_l:
	delta_finish_HALF_3CP_l.append((st - start_host).total_seconds()*1000)


## NOW READ AND PARSE THE POWER TRACE
df = pd.read_csv(reportSmi)
df = df.rename(columns={' power.draw [W]': ' power'})

df = df.drop(' name',axis=1)
df = df.drop(' driver_version',axis=1)
df = df.drop(' pstate',axis=1)
df[' power'] = df[' power'].str.replace(" W", "")
df[[' power']] = df[[' power']].apply(pd.to_numeric)
df['timestamp'] = df['timestamp'].str.replace(r'\d+/\d+/\d+ ', '')
df['timestamp'] = pd.to_datetime(df['timestamp'])

new_df = df.copy()
start_host = start_host.replace( year=new_df['timestamp'].iloc[0].year, month=new_df['timestamp'].iloc[0].month, day=new_df['timestamp'].iloc[0].day)
# print(new_df)

# transform all timestamps into deltas in relation to host started, and convert into miliseconds (*1000)
new_df['timestamp'] = new_df['timestamp']-start_host
new_df['timestamp'] = new_df['timestamp'].dt.total_seconds()*1000

# get the average power between host start and read end
df_one_cycle = new_df[(new_df['timestamp']>=delta_start_write_l[0]) & (new_df['timestamp']<=delta_finish_HALF_3CP_l[-1])]
avg_power = df_one_cycle[' power'].mean()
print('AVERAGE POWER ' + str(avg_power))
#print('AVERAGE POWER %d' % (avg_power))

ActiveGpuTime = (finish_HALF_3CP_l[-1] - start_write_l[0]).total_seconds()*1000 #in ms
print('Timespan between WRITE_START and FINISH_HALF_3CP %dms' % (ActiveGpuTime))
energy = (avg_power*ActiveGpuTime*0.001)
print('Total energy of a cycle between WRITE_START and READ_FINISH %f' % energy)

if(PLOT):
	plt.plot(new_df['timestamp'], new_df[' power'], linestyle='-', linewidth=5)
	plt.plot()
	plt.xlabel('ms since host start')
	plt.ylabel('power [W]')
	#title = sys.argv[2].split('.')[0] + ' Avg Power=' + str(avg_power) + 'W ActiveGpuTime=' + str(ActiveGpuTime) + 'ms Energy=' + str(energy) + ' Joules'
	title = '%s AvgPower=%.2fW ActiveGpuTime=%.2fms Energy=%.2fJ' % (sys.argv[2].split('.')[0], avg_power, ActiveGpuTime, energy)
	plt.title(title)
	plt.xlim([0,1.1*delta_finish_HALF_3CP_l[-1]])
	# draw vertical lines for start/end of operations

	for p in delta_start_write_l:
		plt.axvline(x=p, c='r', label="Start a frame") 

	plt.axvline(x=delta_finish_HALF_3CP_l[-1], c='r', linestyle='-.', label="Finish last frame") 

	plt.legend()	




	plt.plot([delta_start_write_l[0], delta_finish_HALF_3CP_l[-1]],[avg_power, avg_power], 'go', linestyle='--')
	plt.show()