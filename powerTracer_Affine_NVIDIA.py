import os
import sys
from multiprocessing import Process, Event
import subprocess
import signal


def run_smi(event, traceName):
	proc = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,driver_version,pstate,power.draw", "--format=csv", "-lms", "1", "--filename=%s.csv" % (traceName), "&"])
	print("TRACE %s " % (traceName))
	#print("SMI PID %d" % (proc.pid))
	while not event.is_set():
		pass

	# Wait for a moment and kill nvidia-smi
	os.system("sleep 0.2")
	os.kill(proc.pid, signal.SIGTERM) 
	

def run_kernel(event, exe, origSamples, refSamples, affineCosts, nFrames, reportName):
	cmd = "%s GPU 0 22 %s %s %s %s > %s.txt" % (exe, origSamples, refSamples, affineCosts, nFrames, reportName)
	# print(cmd)
	os.system(cmd)
	#os.system("sleep 2")
	event.set()



if __name__=='__main__':

	# 1080
	exe = "./main_1080"
	refSamples = "data/rec_0_QP22_Bask_BQTer_Cac_Mark_Rit_65f.csv"
	origSamples = "data/orig_1_QP22_Bask_BQTer_Cac_Mark_Rit_65f.csv"
	affineCosts_preffix = "Outputs_Tese/QP22_Bask_BQTer_Cac_Mark_Rit_x2_"
	report_preffix = "Outputs_Tese/Report_1080_nFrames"
	trace_preffix = "Outputs_Tese/PowerTrace_1080_nFrames"

	for n in range(5,6):
		for f in range(5):
			nFrames = str(n)
			rep = str(f)
			affineCosts = affineCosts_preffix + nFrames + '_rep' + rep
			report = report_preffix + nFrames + '_rep' + rep
			trace = trace_preffix + nFrames + '_rep' + rep
			print(trace)

			event = Event() # the event is unset when created

			p1 = Process(target=run_smi, args=(event, trace, ))
			p1.start()
			p2 = Process(target=run_kernel, args=(event, exe, origSamples, refSamples, affineCosts, nFrames, report, ))
			p2.start()
			
			p1.join()
			p2.join()
		