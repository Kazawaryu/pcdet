import atexit
import pynvml
import time
import subprocess

def run_next_task():
    time.sleep(20)
    subprocess.run(['bash', "./scripts/dist_train_vr.sh", "4"])
    
pynvml.nvmlInit()
while True:
    handle = pynvml.nvmlDeviceGetHandleByIndex(2)    
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if meminfo.free / 1024**2 > 10000:
        subprocess.run(['bash', "./scripts/dist_train_pp.sh" , "4"])
        atexit.register(run_next_task)
    else:
        print("{} was used, waiting for 30 seconds".format(meminfo.used / 1024**2))
        time.sleep(30)




