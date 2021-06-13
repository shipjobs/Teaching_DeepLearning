from VanillaNet.RNN import *
from VanillaNet.GRU import *
from VanillaNet.LSTM import *
from VanillaNet.CNN import *
from VanillaNet.GAN import *
from VanillaNet.DQN import *

from util import *     #File I/O , log 등 수행
from DataMng import *  #각 모델에 요구되는 Data 구성 


###########################################################################
#작성자 : insung-lee (https://shippauljobs.blogspot.com/)
#작성일 : 2021-06-11 ~ 
#목  적 : 임직원 대상 Deep Learning 모델별 Vanilla Netwotk 을  구성하여 각 모델의 특징을 이해 하는돕고자 함.
#내  용 : Readme.md 참조
#개발 방향
#   - Auto DeepLearning Model 학습기 개발 프로젝트
#   - (기초) 강의/학습/실습을 목적으로 잘알려진 모델을 대상으로 함
###########################################################################

while True:
    print()
    print("-----------------------------------------------------")
    print("------ShipJob DL 모델 학습기 (insung-Lee)-------------")   
    print("[1]. RNN  [2]. GRU  [3]. LSTM")  
    print("[4]. CNN  [5]. GAN  ")  
    print("[6]. DQN  ") 
    print("[7]. Exit")
    print("-----------------------------------------------------------")
   
    menu = int(input(">>>>> select: "))
    if menu   == 1:
        print(">>>>> Selected RNN Model <<<<<")  # 시계열
        # 학습 데이터 요청 선택
        # RNN 모델 지정..
        # RNN 학습 수행
        # RNN 학습 현황 가시화
        # RNN 결과 보기  , 그래프 보기
        # 학습 이미지 출력?(저장)
        
        pass
    elif menu == 2:
        print(">>>>> Selected GRU Model <<<<<")  #  "  개선
        pass    
    elif menu == 3:
        print(">>>>> Selected LSTM Model <<<<<") #  "  개선 2
        pass    
    elif menu == 4:
        print(">>>>> Selected CNN Model <<<<<")  # 합성곱
        pass    
    elif menu == 5:
        print(">>>>> Selected GAN Model <<<<<")  # 적대적 생성적
        pass    
    elif menu == 6:
        print(">>>>> Selected DQN Model <<<<<")  # 강화 학습
        pass    
    elif menu == 7:
        print(">>>>> Exit <<<<<")  # 강화 학습
        break   
    else:
        print(">>>>> wrong input!")


