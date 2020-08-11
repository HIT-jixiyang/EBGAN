import numpy as np
import netCDF4
from netCDF4 import Dataset

def get_aws_x_y(lon,lat):
    sourth_west=(108.505,19.0419)
    north_east=(117.505,26.0419)
    delta_x=(north_east[0]-sourth_west[0])/900
    delta_y=(north_east[1]-sourth_west[1])/700
    position_x = int(900 * ((lon - 108.505) / (117.505 - 108.505)))
    position_y = int(700 * ((lat - 19.0419) / (26.0419 - 19.0419)))
    left_up_i = []
    left_bottom_i = []
    right_up_i = []
    right_bottom_i = []
    for i in range(position_x-2,position_x+2):
        for j in range(position_y-2,position_y+2):
            if lon-sourth_west[0]>=i*delta_x and lon-sourth_west[0]<=(i+1)*delta_x and lat-sourth_west[1]>=j*delta_y and lat-sourth_west[1]<=(j+1)*delta_y:
                left_up_i = [i,j+1]
                left_bottom_i = [i,j]
                right_up_i=[i+1,j+1]
                right_bottom_i=[i+1,j]
                break
    left_up=(left_up_i[0]*delta_x,left_up_i[1]*delta_y)
    left_bottom=(left_bottom_i[0]*delta_x,left_bottom_i[1]*delta_y)
    right_up=(right_up_i[0]*delta_x,right_up_i[1]*delta_y)
    right_bottom=(right_bottom_i[0]*delta_x,right_bottom_i[1]*delta_y)
    d1=get_distance(lon,left_up[0],lat,left_up[1])
    d2=get_distance(lon,left_bottom[0],lat,left_bottom[1])
    d3=get_distance(lon,right_up[0],lat,right_up[1])
    d4=get_distance(lon,right_bottom[0],lat,right_bottom[1])
    if min([d1,d2,d3,d4])==d1:
        return left_up_i
    elif min([d1,d2,d3,d4])==d2:
        return left_bottom_i
    elif min([d1, d2, d3, d4]) == d3:
        return right_up_i
    elif min([d1,d2,d3,d4])==d4:
        return right_bottom_i


def get_distance(x1,y1,x2,y2):
    return (x1-x2)**2+(y1-y2)**2


nc_obj=Dataset('/home/ices/work/jxy/gru_tf_5_layer_5_3/FACT_HOUR_20190411.nc')
v=nc_obj.variables
lat=v['lat'][:].data
lon=v['lon'][:].data
rain_hour=v['hourr'][:].data
time=v['time'][:].data
rain_21=rain_hour[21]
rain_22=rain_hour[22]
qpe=np.loadtxt('/home/ices/work/jxy/gru_tf_5_layer_7_5_3_no_relu/201904112100_qpe.txt',dtype=np.float32)
x_y=[]
# position_x=[]
# position_y=[]
# for i in range(len(lat)):
#     x,y=get_aws_x_y(lon[i],lat[i])
#     position_x.append(x)
#     position_y.append(700-y)
# print(lon[20],lat[20],position_x[20],position_y[20])

position_x= 900 * ((lon - 108.505) / (117.505 - 108.505))
position_y = 700 * ((lat- 19.0419) / (26.0419 - 19.0419))
position_y=700-position_y
position_x=position_x.astype(int)
position_y=position_y.astype(int)
#
# id=v['obtid'][:].data
# w=open('/home/ices/work/jxy/gru_tf_5_layer_5_3/201904112100.txt','w')
# for i in range(0,2302):
#     if x_y[i][0]>=300 and x_y[i][0]<=600 and x_y[i][1]>=200 and x_y[i][1]<=500:
#         w.write(str(x_y[i][0]) + ',' + str(x_y[i][1]) + ',' + str(rain_21[i]) + ',' + str(
#             rain_22[i]) + '\n')
# w.close()
#
import cv2
rain_map=np.zeros([700,900])
for i in range(len(lat)):
    if (position_x[i]==667 and position_y[i]==241) or (position_x[i]==464 and position_y[i]==341):

        rain_map[position_y[i]-5:position_y[i]+5,position_x[i]-5:position_x[i]+5]=255
    else:
        rain_map[position_y[i]-2:position_y[i]+2,position_x[i]-2:position_x[i]+2]=rain_21[i]
cv2.imwrite('/extend/rain_map.png',rain_map)

error=[]
error1=[]
error1_10=[]
error10_20=[]
error_20_=[]
k=0
rain_aws=0
for i in range(0,len(lat)):
    # if position_x[i] >= 300 and position_x[i] <= 600 and position_y[i] >= 200 and position_y[i] <= 500:
        rain=rain_21[i]
        if rain>0:
            rain_aws=rain_aws+1
        if rain<0:
            continue
        q=qpe[position_y[i]][position_x[i]]
        if rain<1:
            error1.append(np.abs(q-rain))
            if np.abs(q - rain) > 20:
                print('0-1',position_x[i], position_y[i],q)
                k=k+1
        elif rain<10:
            error1_10.append(np.abs(q-rain))
            if np.abs(q-rain)>20:
                print(position_x[i],position_y[i])
                k = k + 1
        elif rain <20:
            error10_20.append(np.abs(q-rain))
            if np.abs(q-rain)>20:
                print(position_x[i],position_y[i])
                k = k + 1
        else:
            error_20_.append(np.abs(q-rain))
            if np.abs(q - rain) > 20:
                print(position_x[i], position_y[i])
                k = k + 1
        error.append(np.abs(q-rain))
print('k',k,'共有',rain_aws)
error1=np.array(error1)
error_20_=np.array(error_20_)
error10_20=np.array(error10_20)
error1_10=np.array(error1_10)
error=np.array(error)
print('整体平均绝对误差',np.mean(error),'共{}个点'.format(len(error)),'最大误差',np.max(error),len(error[error>20]))
print('0-1误差',np.mean(error1),'共{}个点'.format(len(error1)),'最大误差',np.max(error1),len(error1[error1>20]))
print('1-10误差',np.mean(error1_10),'共{}个点'.format(len(error1_10)),'最大误差',np.max(error1_10),len(error1_10[error1_10>20]))
print('10-20误差',np.mean(error10_20),'共{}个点'.format(len(error10_20)),'最大误差',np.max(error10_20),len(error10_20[error10_20>20]))
print('20-误差',np.mean(error_20_),'共{}个点'.format(len(error_20_)),'最大误差',np.max(error_20_),len(error_20_[error_20_>20]))
from matplotlib import pyplot as plt
plt.hist(error.ravel(),50,[1,50])
plt.show()
