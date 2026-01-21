### Sample function here:

```python
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
```

type of command that is handled by this `cmd = "nvidia-smi --format=csv --query-gpu=power.draw,fan.speed,utilization.gpu,temperature.gpu,memory.used,memory.total"`
