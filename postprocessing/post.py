import os
from color_map import multi_process_transfer
import pandas as pd
import os
import numpy as np
from PIL import Image
import shutil as sh
from evaluation import Evaluator
evaluator = Evaluator(os.path.join('/extend/gru_tf_data/5l_stlstm_incept_ebgan/Test/199999-metrics-hist'))
path = '/extend/gru_tf_data/5l_stlstm_incept_ebgan/Test/199999-2029'
time = ['201903010806', '201910011200']
time_range = pd.date_range(start=time[0], end=time[1], freq='6min')
for i in range(len(time_range)):
    date = time_range[i].strftime("%Y%m%d%H%M")
    save_path = os.path.join(path, date)

    if os.path.exists(save_path):
        # try:
        #
        #     sh.copytree(save_path + '/pred', save_path + '/origin')
        # except:
        #     pass
        target_png=os.path.join(save_path,'in','5.png')
        des_png=os.path.join(save_path,'in','10.png')
        if not os.path.exists(des_png):
            os.rename(target_png,des_png)
        os.system(r'./postprocessing' + ' ' + os.path.join(save_path))
        display_path=os.path.join('/extend/gru_tf_data/5l_stlstm_incept_ebgan/','display/199999-eval/',date)
        save_path=os.path.join(path,date)
        multi_process_transfer(save_path+'/in',display_path+'/in')
        multi_process_transfer(save_path+'/out',display_path+'/out')
        multi_process_transfer(save_path+'/pred',display_path+'/pred')
        # multi_process_transfer(save_path+'/origin',display_path+'/origin')
        print('done',save_path)