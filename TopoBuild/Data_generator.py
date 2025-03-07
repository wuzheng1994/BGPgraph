import time

# second time transfer to '%Y-%m-%d %H:%M:%S'.
def s2t(seconds:int) -> str:
    utcTime = time.gmtime(seconds)
    strTime = time.strftime("%Y-%m-%d %H:%M:%S",utcTime)
    return strTime

# str time transfer to second time.
def t2s(str_time:str) -> int:
    time_format = '%Y-%m-%d %H:%M:%S'
    time_int = int(time.mktime(time.strptime(str_time, time_format)))
    return time_int

# generate the update traffic within the given windows with labels.
def data_generator_wlabel(files, Period, start_time:str, end_time:str, anomaly_start_time:str, anomaly_end_time:str):
    if files == []:
        print("The updates files is null, please check!")
        
    updates_list = []
    interval = Period * 60
    left_time = t2s(start_time)
    right_time = left_time + interval
    end_time = t2s(end_time)

    anomaly_start_time = t2s(anomaly_start_time)
    anomaly_end_time = t2s(anomaly_end_time)

    count = 0
    for file in files:
        with open(file) as f:
            for l in f:
                if l.strip() != '':
                    line = l.strip().split('|')
                    time_ = line[1]
                    prefix_ = line[5] 
                    if '.' in prefix_: # 只接受Ipv4的前缀  
                        if '.' not in time_:
                            time_ = int(time_)
                        else:
                            time_ = int(float(time_))
                        if time_ < left_time:
                            pass
                        
                        elif time_ >= left_time and time_ <= right_time:
                            # line: 定义line的内容
                            updates_list.append(line)
                        
                        elif time_ > right_time and time_ <= end_time:
                            if count % 100 == 0:
                                print('No.{}: the starting time {} and ending time {}'.format(count, s2t(left_time).split(' ')[1],s2t(right_time).split(' ')[1]))
                            left_time = right_time
                            right_time += interval
                            count += 1                            
                            if right_time < anomaly_start_time or left_time > anomaly_end_time:
                                yield (updates_list,'0') # normal label
                            else:
                                yield (updates_list, '1') # anomaly label
                            updates_list = []
                        elif time_ > end_time:
                            break

# generate the update traffic within the given windows with no label. 
# Aadopted in unsupervised methods.
def data_generator_wolabel(files, Period, start_time:str, end_time:str):
    updates_list = []
    interval = Period * 60
    left_time = t2s(start_time)
    right_time = left_time + interval
    end_time = t2s(end_time)
    count = 0
    for file in files:
        with open(file) as f:
            for l in f:
                if l.strip() != '':
                    line = l.strip().split('|')
                    time_ = line[1]
                    prefix_ = line[5] 
                    if '.' in prefix_: # 只接受Ipv4的前缀  
                        if '.' not in time_:
                            time_ = int(time_)
                        else:
                            time_ = int(float(time_))
                        if time_ <= right_time:
                            # line: 定义line的内容
                            updates_list.append(line)
                        elif time_ > right_time and time_ <= end_time:
                            if count % 100 == 0:
                                print('No.{}: the starting time {} and ending time {}'.format(count, s2t(left_time).split(' ')[1],s2t(right_time).split(' ')[1]))
                            left_time = right_time
                            right_time += interval
                            count += 1                            
                            yield (updates_list,None) # normal label
                            updates_list = []
                        elif time_ > end_time:
                            break
