
import numpy as np


from torch.utils.data import Dataset

class MetaField:
    iid = 0
    width = 1
    height = 2
    initialFrame = 3
    finalFrame = 4
    numFrames = 5
    class_ = 6
    drivingDirection = 7
    traveledDistance = 8
    minXVelocity = 9
    maxXVelocity = 10
    meanXVelocity = 11
    minDHW = 12
    minTHW = 13
    minTTC = 14
    numLaneChanges = 15

class Field:
    frame = 0
    iid = 1
    x = 2
    y = 3
    width = 4
    height = 5
    xVelocity = 6
    yVelocity = 7
    xAcceleration = 8
    yAcceleration = 9
    frontSightDistance = 10
    backSightDistance = 11
    dhw = 12
    thw = 13
    ttc = 14
    precedingXVelocity = 15
    precedingId = 16
    followingId = 17
    leftPrecedingId = 18
    leftAlongsideId = 19
    leftFollowingId = 20
    rightPrecedingId = 21
    rightAlongsideId = 22
    rightFollowingId = 23
    laneId = 24

    @staticmethod
    def set_zero_frame(item):
        item[Field.x] = 0
        item[Field.y] = 0
        item[Field.xAcceleration] = 0
        item[Field.xVelocity] = 0
        item[Field.yVelocity] = 0

import torch
@torch.no_grad()
def classify_longitudinal_intention(trajectory_data, threshold=1):
    """
    根据车辆轨迹数据分类整体纵向意图
    
    参数:
        trajectory_data (torch.Tensor): Tx6 的车辆轨迹张量 [x, y, xV, yV, xA, yA]
        threshold (float): 速度变化阈值(m/s)，默认0.1
        
    返回:
        torch.Tensor: 意图标签 [ACC, DEC, CON]
    """
    trajectory_data = torch.tensor(trajectory_data)
    # 提取速度分量 (xV, yV)
    velocities = trajectory_data[:, 4] + trajectory_data[:, 5]
    
    # 计算各时刻的速度大小 (标量速度)
    speed = velocities.mean()

    # 根据速度变化分类意图
    if speed > threshold:
        return 0  # ACC
    elif speed < 0:
        return 1  # DEC
    else:
        return 2  # CON

def handle_csv(filename):
    import csv

    # 假设数据存储在名为 'data.txt' 的文件中

    # 读取文件并将数据存储为列表
    data_list = []

    with open(filename, newline='') as file:
        reader = csv.reader(file, delimiter=',')  # 使用tab作为分隔符
        for row in reader:
            data_list.append(row)

    res = []
    for rowi in range(1,len(data_list)):
        res.append([float(i) for i in data_list[rowi]])

    return res

def handle_csv_meta(filename):
    import csv

    # 假设数据存储在名为 'data.txt' 的文件中

    # 读取文件并将数据存储为列表
    data_list = []

    with open(filename, newline='') as file:
        reader = csv.reader(file, delimiter=',')  # 使用tab作为分隔符
        for row in reader:
            data_list.append(row)

    res = {}
    for rowi in range(1,len(data_list)):
        res[int(data_list[rowi][0])] = int(data_list[rowi][MetaField.drivingDirection])

    return res

class HighD(Dataset):

    def __init__(self):
        super().__init__()
        self.source = []

    def expend(self,pkl):
        self.source.extend(pkl)

def todict(source):
    curs = {}
    cursf = {}
    cur_cid = source[0][Field.iid]
    p = 0
    for i,row in enumerate(source):
        if row[Field.iid] != cur_cid:
            curs[cur_cid] = source[p:i]
            cursf[cur_cid] = {}
            for j in source[p:i]:
                cursf[cur_cid][j[Field.frame]] = j
            p = i
            cur_cid = row[1]
    return curs,cursf

def gety(driving_dir,cur_lane,start_lane):
    if driving_dir == 1:
        if  cur_lane < start_lane:
            label = 2 # right lane change
        elif cur_lane > start_lane:
            label = 0 # left lane change                      
    elif driving_dir == 2:
        if cur_lane > start_lane:
            label = 2 # right lane change
        elif cur_lane < start_lane:
            label = 0 # left lane change
    return label

def get_x(x,frameInfoDct):
    """
        precedingId = 16
        followingId = 17
        leftPrecedingId = 18
        leftAlongsideId = 19
        leftFollowingId = 20
        rightPrecedingId = 21
        rightAlongsideId = 22
        rightFollowingId = 23
    """
    # x = 2
    # y = 3
    # xVelocity = 6
    # yVelocity = 7
    auxInfoIds = [2,3,6,7]
    extend = {i:[] for i in range(len(x))}
    
    for i in range(16,24):
        for j,xi in enumerate(x):
            if xi[i] == 0 or xi[i] not in frameInfoDct:
                tend = [0,0,0,0]
            else:
                frame  = xi[Field.frame]
                acurId = xi[i]
                # tend = [frameInfoDct[acurId][frame][j] for j in auxInfoIds]
                tend = [frameInfoDct[acurId][frame][j] - xi[j] for j in auxInfoIds]
            extend[j].extend(tend)
    
    for i in extend:
        neighboring_vehicles = []
        low = extend[i]
        for j in range(0,len(extend[i]),4):
            if sum(low[j:j+4]):
                neighboring_vehicles.append(1)
            else:
                neighboring_vehicles.append(0)
        neighboring_vehicles.insert(4,0)
        extend[i].extend(neighboring_vehicles)

    keepid = [2,3,6,7,8,9]
    res = []
    for i in range(len(x)-1,-1,-1):
        x[i][Field.x] = x[i][Field.x] - x[0][Field.x]
        x[i][Field.y] = x[i][Field.y] - x[0][Field.y]
        x[i][Field.xVelocity] = x[i][Field.xVelocity] - x[0][Field.xVelocity]
        x[i][Field.yVelocity] = x[i][Field.yVelocity] - x[0][Field.yVelocity]
        x[i][Field.xAcceleration] = x[i][Field.xAcceleration] - x[0][Field.xAcceleration]
        x[i][Field.yAcceleration] = x[i][Field.yAcceleration] - x[0][Field.yAcceleration]
        res.append([x[i][j] for j in keepid] + extend[i])

    # data : [x,y,xV,yV,xA,yA]:6 + [Δx,Δy,ΔxV,ΔyV for i in [16 - 23]]:4x8 + [neighboring_vehicles \in R^(3x3)] 

    return res[::-1]

def extract_dist(arr):
    negatives = []
    positives = []
    
    # 遍历列表，分别收集负数和正数
    for item in arr:
        num = item[1]
        if num < 0:
            if len(negatives) < 6:
                negatives.append(item[0])
        elif num > 0:
            if len(positives) < 6:
                positives.append(item[0])
        
        # 如果已经收集到足够的负数和正数，提前终止循环
        if len(negatives) == 6 and len(positives) == 6:
            break
    
    # 补足6个负数（不足部分用0填充）
    negatives += [-1] * (6 - len(negatives))
    # 补足6个正数（不足部分用0填充）
    positives += [-1] * (6 - len(positives))
    
    # 合并结果（先负数后正数）
    return negatives + [-1] + positives

def Dir(v1, p1, p0):
    # 计算向量 p1 - p0
    v2 = [p1[0] - p0[0], p1[1] - p0[1]]
    
    # 计算点积：v1[0]*v2[0] + v1[1]*v2[1]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # 根据点积判断夹角
    return 1 if dot_product > 0 else -1

def Distant(p1,p2):
    return ((p1[0]- p2[0])**2 + (p1[1]- p2[1])**2)**0.5 

def get_x_with_social(x,frameInfoDct):
    """
        precedingId = 16
        followingId = 17
        leftPrecedingId = 18
        leftAlongsideId = 19
        leftFollowingId = 20
        rightPrecedingId = 21
        rightAlongsideId = 22
        rightFollowingId = 23
    """
    # x = 2
    # y = 3
    # xVelocity = 6
    # yVelocity = 7
    auxInfoIds = [2,3,6,7]
    extend = {i:[] for i in range(len(x))}
    

    reg = [x[-1][Field.x] - x[0][Field.x],x[-1][Field.y] - x[0][Field.y]]
    #选取前后最近的6辆车 6 + 1 + 6
    for j,xi in enumerate(x):
        laneId = xi[Field.laneId]
        frame = xi[Field.frame]
        frame_pos = [xi[Field.x],xi[Field.y]]
        dist = []
        for car_id in frameInfoDct:
            if frame not in frameInfoDct[car_id]:
                continue
            fi = frameInfoDct[car_id][frame]
            if frame in fi:
                pos = [fi[Field.x],fi[Field.y]]
                dist.append((car_id, Dir(reg,pos,frame_pos) * Distant(frame_pos,pos)))
        dist.sort(key=lambda item:item[1])
        cars = extract_dist(dist)
        
        lanes = []

        for car_id in cars:
            if car_id < 0:
                lanes.append(-1)
                tend = [0,0,0,0]
            else:
                temp_laneId = frameInfoDct[car_id][frame][Field.laneId]
                if temp_laneId - laneId > 0:
                    lanes.append(0)
                elif temp_laneId - laneId == 0:
                    lanes.append(1)
                else:
                    lanes.append(2)
                tend = [frameInfoDct[car_id][frame][j] - xi[j] for j in auxInfoIds]
            extend[j].extend(tend)

        extend[j].extend(lanes)

    keepid = [2,3,6,7,8,9]
    x = torch.tensor(x)[:,keepid] 
    x = (x - x[0]).tolist()
    
    res = []
    for i in range(len(x)):
        res.append(x[i] + extend[i])

    # data : [x,y,xV,yV,xA,yA]:6 + [Δx,Δy,ΔxV,ΔyV for i in [16 - 23]]:4x13 + [neighboring_vehicles \in R^13] 

    return res

def infere_data(res):
    if len(res) < 2:
        return False
    direction = res[1][0] - res[0][0]
    for i in range(2,len(res)):
        if (res[i][0] - res[0][0]) * direction <= 0:
            return True
    return False

def todata(dct:dict,meta,frameInfoDct,tho = 3,pre_tho = 5):
    if tho > 0:
        framelen = int(tho * 25)
    else:
        framelen = 1
    pfl = int(pre_tho * 25)
    data = []
    for key in dct:
        item = dct[key]
        if len(item) < framelen:
            continue
        ddir = meta[key]
        plane = item[0][-1]
        pi = 0
        lastkeepi = 0
        keep_count = 0
        for i in range(len(item)):
            curlane = item[i][-1]
            if curlane != plane:
                x = item[max(pi,i - framelen):i]
                # if len(x) == framelen:
                data.append((
                    get_x_with_social(x[::5],frameInfoDct),
                    (torch.tensor(item[i:i+pfl:5]) - torch.tensor(x[0]))[:,[Field.x,Field.y]].tolist(),
                    gety(ddir,item[i][Field.laneId],plane),
                    classify_longitudinal_intention(x)
                ))
                plane = curlane
                pi = i
            elif curlane == plane and (i - framelen) > pi and (i - lastkeepi) >= framelen and keep_count < 1:
                x = item[i-framelen:i]
                data.append((
                    get_x_with_social(x[::5],frameInfoDct),
                    (torch.tensor(item[i:i+pfl:5]) - torch.tensor(x[0]))[:,[Field.x,Field.y]].tolist(),
                    1,
                    classify_longitudinal_intention(x)
                ))
                lastkeepi = i
                keep_count += 1

    return data


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_file(i):
    # i 是 int，这里统一补零处理
    file_index = f"{i:02d}"
    meta = handle_csv_meta(f'./highD-dataset-v1.0/data/{file_index}_tracksMeta.csv')
    data = handle_csv(f'./highD-dataset-v1.0/data/{file_index}_tracks.csv')
    dct, dctf = todict(data)
    return todata(dct, meta, dctf, tho=3)


if __name__ == '__main__':

    from tqdm.auto import tqdm
    from random import shuffle,seed
    seed(42)

    for T in [3]: # ,8,12,14,16,18,20
        
        result = []
        with ProcessPoolExecutor(max_workers=10) as executor:
            for output in tqdm(executor.map(process_file, range(1, 61)), total=60):
                result.extend(output)

        result = [data for data in result if len(data[0]) == 3 * 5]
        from collections import Counter
        cres = Counter([i[2] for i in result])
        print(cres)

        blanced_result,count = [],0
        for i in result:
            if i[2] == 1 and count < (cres[0] + cres[2])/2:
                blanced_result.append(i)
                count += 1
            elif i[2] != 1:
                blanced_result.append(i)

        shuffle(blanced_result)

        from collections import Counter
        cres = Counter([i[2] for i in blanced_result])
        print(cres)

        cutline = int(len(blanced_result) * 0.8)

        with open(f'./datasets/highD_{T}_train.txt','w+') as fp:
            fp.write(str(blanced_result[:cutline]))
        
        with open(f'./datasets/highD_{T}_test.txt','w+') as fp:
            fp.write(str(blanced_result[cutline:]))


