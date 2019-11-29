import numpy as np
import json




def conv2d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None,
           data_format="NCHW"):



    
