import subprocess
import pandas as pd
import psutil
#from io import StringIO
#import csv
from datetime import datetime

def cpu_util_metrics():
    """
        Function to return CPU usage for the system at a specific timestamp.
        Returns:
            Pandas DataFrame of the metrics
    """
    #with open(StringIO(), 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(["Timestamp", "CPU Usage", "Memory Usage", "Disk Usage"])
    while True:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #    writer.writerow([timestamp, cpu, memory, disk])
        #    file.flush()  # Ensure data is written to file    
        print(f"{timestamp} - CPU: {cpu}%, Memory: {memory}%, Disk: {disk}%")
        #time.sleep(60)  # Log every minute
    return 

def data_entry_clean(line):
    """
        Function is for iterating over a split line and remove white spaces
        at the front and back of the string.
    """
    return [entry.strip() for entry in line.split(",")]

def gpu_util_metrics2df(cmd: None) -> pd.DataFrame:
    """
        Collects GPU metrics for the system.
        Returns:
            GPU metrics represented as Pandas DataFrame
    """
    if cmd is not None:
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            ## the above returns timestamp at that specific moment
            for k,line in enumerate(p.stdout.readlines()):
                line = line.decode().strip("\n")
                if k==0:
                    columns_list = data_entry_clean(line)
                elif k==1:
                    data_dictionary = {col:[] for col in columns_list}
                    data=data_entry_clean(line)
                    for i,value in enumerate(data):
                        data_dictionary[columns_list[i]].append(value)
                    df = pd.DataFrame.from_dict(data_dictionary)
                ## add the timestamp to these metrics
            df["Timestamp"] = timestamp
            return df
        except Exception as e:
            print("command is invalid \n", e)
def main():
    #print(cpu_util_metrics())
    gpu_cmd="nvidia-smi --format=csv --query-gpu=power.draw,fan.speed,utilization.gpu,temperature.gpu,memory.used,memory.total"
    print(gpu_util_metrics2df(gpu_cmd))
    #print(gpu_util_metrics2df())
if __name__ == '__main__':
    main()
