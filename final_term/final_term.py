import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D

def read_data(filename):
    data=pd.read_csv(filename)
    return data
    
        
if __name__ == '__main__':
    #데이터 로드
    df=read_data("data/nba2021_advanced.csv")
    
    #모델 트레이닝
    plt.title(f'NBA')
    x=df["PER"]
    y=df["WS/48"]
    line_fitter = LinearRegression()
    line_fitter.fit(x.values.reshape(-1,1), y)
    plt.plot(x,y,'o')
    plt.plot(x,line_fitter.predict(x.values.reshape(-1,1)))
    plt.show()
    plt.xlabel('PER')  # x축 아래에 레이블을 그린다.
    plt.ylabel('WS/48')