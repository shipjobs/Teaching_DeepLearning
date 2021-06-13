from numpy.core.fromnumeric import shape
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

#######################################################################
# One-hot 벡터로 변환
#######################################################################
def string_to_onehot(string, len_vector_size ):

    start = np.zeros(shape= len_vector_size , dtype=int)
    end = np.zeros(shape= len_vector_size , dtype=int)
    start[-2] = 1
    end[-1] = 1

    #문자열의 문자들을 차례대로 받아서 진행
    for i in string:
        
        idx = char_list.index(i)  #문장내 해당 문자의 인덱스(순서)
        zero = np.zero(shape = n_letters , dtype = int)  # 0 으로 구성된 배열 만들기
        zero[idx] = 1  # 해당 문자 인덱스만 1로 변경

        start = np.vstack([start , zero])  #start 와 새로 생긴 zero 를 붙이고 이를 start 에 할당
    
    output = np.vstack([ start , end ])

    return output


# One-hot 벡터를 문자로 바꿔주는 함수 
# [1 0 0 ... 0 0] -> a 
def onehot_to_word(onehot_vector):
    onehot = torch.tensor.numpy(onehot_vector)
    return char_list[onehot.argmax()]
