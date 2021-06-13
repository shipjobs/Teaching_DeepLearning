"""
from numpy.core.fromnumeric import shape
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
"""

from util.sentence_edit import *

# 하이퍼파라미터 설정
n_hidden = 35   # Hidden layer 개수
lr = 0.01       # Learning late (학습률)
epochs = 1000   # 반복 학습

############## 임시  (추후,, 파일에서 읽어 오도록 수정.. ) ##############
# 사용하는 문자는 영어 소문자 및 몇가지 특수문자
# alphabet(0-25), space(26), ... , start(0), end(1)
string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
chars =  "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
#######################################################################

# char 의 문자들을 리스트로변경. 
char_list = [i for i in chars]

# 길이(=문자의 개수)를 저장
n_letters = len(char_list)


# RNN with 1 hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.act_fn = nn.Tanh()
    
    def forward(self, input, hidden):
        # 입력과 hidden state를 cat함수로 붙여줍니다.
        combined = torch.cat((input, hidden), 1)
        # 붙인 값을 i2h 및 i2o에 통과시켜 hidden state는 업데이트, 결과값은 계산해줍니다.
        hidden = self.act_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden
    
    # 아직 입력이 없을때(t=0)의 hidden state를 초기화해줍니다. 
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    

rnn = RNN(n_letters, n_hidden, n_letters)


# 손실함수와 최적화함수를 설정해줍니다.

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


########################################################################################
# train

# 문자열을 onehot 벡터로 만들고 이를 토치 텐서로 바꿔줍니다.
# 또한 데이터타입도 학습에 맞게 바꿔줍니다.
one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())

for i in range(epochs):
    optimizer.zero_grad()
    # 학습에 앞서 hidden state를 초기화해줍니다.
    hidden = rnn.init_hidden()
    
    # 문자열 전체에 대한 손실을 구하기 위해 total_loss라는 변수를 만들어줍니다. 
    total_loss = 0
    for j in range(one_hot.size()[0]-1):
        # 입력은 앞에 글자 
        # pyotrch 에서 p y t o r c
        input_ = one_hot[j:j+1,:]
        # 목표값은 뒤에 글자
        # pytorch 에서 y t o r c h
        target = one_hot[j+1]
        output, hidden = rnn.forward(input_, hidden)
        
        loss = loss_func(output.view(-1),target.view(-1))
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(total_loss)

########################################################################################
# train
# test 
# hidden state 는 처음 한번만 초기화해줍니다.

start = torch.zeros(1,n_letters)
start[:,-2] = 1

with torch.no_grad():
    hidden = rnn.init_hidden()
    # 처음 입력으로 start token을 전달해줍니다.
    input_ = start
    # output string에 문자들을 계속 붙여줍니다.
    output_string = ""

    # 원래는 end token이 나올때 까지 반복하는게 맞으나 끝나지 않아서 string의 길이로 정했습니다.
    for i in range(len(string)):
        output, hidden = rnn.forward(input_, hidden)
        # 결과값을 문자로 바꿔서 output_string에 붙여줍니다.
        output_string += onehot_to_word(output.data)
        # 또한 이번의 결과값이 다음의 입력값이 됩니다.
        input_ = output

print(output_string)