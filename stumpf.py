import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import math


def ChooseBands_Prisma_Hawaii():
    file_location1 = 'data/XXX.csv'
    bands_data = pd.read_csv(file_location1, dtype=float)
    count = len(open(file_location1).readlines())
    n = 10
    R2_max = 0
    band1 = 0
    band2 = 0
    k=0
    b=0
    for i in range(63): # PRISMA:63
        Band1 = np.array(bands_data.iloc[:,4+i])  #csv文件按序号、X、Y、Z、波段顺序，所以从4+i
        for j in range(63):
            if i is not j:
                Band2=np.array(bands_data.iloc[:,4+j])
                Ratio = np.array(np.log(n * Band1) / np.log(n * Band2))
                Ratio = Ratio.reshape(count - 1, 1)
                X = Ratio
                y = np.array(bands_data[['Z']])
                X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.4, random_state=100)
                LR = LinearRegression()
                LR.fit(X_train, y_train)
                R2 = r2_score(y_test, LR.predict(X_test))
                if R2 > R2_max:
                    R2_max = R2
                    band1 = i
                    band2 = j
                    k = LR.coef_
                    b = LR.intercept_
                    reg=LR
                    print('RMSE: %.10f' % math.sqrt(mean_squared_error(y_test, reg.predict(X_test))))
                    print(Band1)
                    print(Band2)
    print(band1 + 4, band2 + 4, R2_max, k, b, n)  #所有+4同上，其他影像修改数字(PRISMA:+4)
    Band_1 = pd.Series(bands_data.iloc[:,band1+4])
    Band_2 = pd.Series(bands_data.iloc[:,band2+4])
    Ratio2 = np.array(np.log(n * Band_1) / np.log(n * Band_2))
    X = Ratio2.reshape(count-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)

    plt.scatter(X_train, y_train, color='orange',label='train')
    plt.scatter(X_test, y_test, color='blue',label='test')
    plt.plot(X, reg.predict(X), 'r-',label='Predict')
    plt.legend()
    plt.xlabel('Ration')
    plt.ylabel('Z')

    f, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 5))
    ax.set_xlim([-30, 5])
    ax.set_ylim([-30, 5])
    Axis_line = np.linspace(*ax.get_xlim(), 2)
    ax.plot(Axis_line, Axis_line, transform=ax.transAxes, linestyle='--', linewidth=2, color='black')  # 1:1标准线
    ax.scatter(y_train, reg.predict(X_train), color="orange", label="Train")
    ax.scatter(y_test, reg.predict(X_test), color="blue", label="Test")
    plt.legend(loc='upper right')
    plt.xticks(fontsize=8, fontweight='normal')
    plt.yticks(fontsize=8, fontweight='normal')
    plt.title('Predictions for Stumpf')
    plt.xlabel('True depth', fontsize=10)
    plt.ylabel('Predicted depth', fontsize=10)
    plt.show()
    #filename = 'D:/Desktop/Stumpf/results/Hawaii_PRI_2m1000_train.txt'
    #np.savetxt(filename, np.column_stack([reg.predict(X_train), y_train]))
    filename = 'results/stumpf/ICESat-2/Hawaii_PRISMA_stumpf_0604.txt'
    np.savetxt(filename, np.column_stack([reg.predict(X_test), y_test]))
    return band1+4, band2+4, k, b, n


def Predict(band1,band2,m1,m0,n):
    file_location = 'data/XXX.csv' #整个影像
    bands_data = pd.read_csv(file_location, dtype=float)
    count = len(open(file_location).readlines())
    R1 = np.array(bands_data.iloc[:,band1-1])
    R2 = np.array(bands_data.iloc[:, band2-1])
    Z_predict = np.array(m1*((np.log(n*R1))/(np.log(n*R2)))-m0)
    Z_predict = Z_predict.reshape(count - 1, 1)
    Local_X = bands_data['X']
    Local_Y = bands_data['Y']
    name = ['X', 'Y', 'Z_predict']
    output = pd.DataFrame()
    output.loc[:, 'X'] = Local_X
    output.loc[:, 'Y'] = Local_Y
    output.loc[:, 'Z_predict'] = Z_predict
    output.to_csv('results/XXX.csv')

if __name__ == '__main__':

    band1, band2, m1, m0, n = ChooseBands_Prisma_Hawaii()
    Predict(band1, band2, m1, -m0, n)


