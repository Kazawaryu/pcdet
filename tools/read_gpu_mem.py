import pynvml

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(meminfo.used / 1024**2)
print(meminfo.free / 1024**2)
