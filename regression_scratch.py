import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("data.csv")

print(df.info())
def loss_function(m,b,data):
    error = 0
    for i in range(len(data)):
        x = data.iloc[i,0]
        y = data.iloc[i,1]
        error += (y-(m*x+b))**2
    return error/float(len(data))
def gradient_decent(m_now,b_now, data,L):
    m_gradient, b_gradient  = 0,0
    n = float(len(data))
    for i in range(len(data)):
        x = data.iloc[i,0]
        y = data.iloc[i,1]
        
        m_gradient += -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient += -(2/n)*(y-(m_now*x+b_now))
    m = m_now - L*m_gradient
    b = b_now - L*b_gradient
    return [m,b]



m = 0
b = 0
L = 0.0001
epochs = 1000



plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title("linear model")
ax2.set_title("loss curve")
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
ax2.set_xlim([0,epochs])
ax2.set_ylim([0,loss_function(m,b,df)])

ax1.scatter(df.iloc[:,0], df.iloc[:,1])
line, = ax1.plot(range(20,80), range(20,80),color="red")
line2, = ax2.plot(0,0)

xlst, ylst = [], []
for i in range(epochs):

    m,b = gradient_decent(m, b, df, L)
    # print(m,b)
    # print("-"*10)
    line.set_ydata(m * range(20,80) + b)
    xlst.append(i)
    ylst.append(loss_function(m, b, df))
    line2.set_xdata(xlst)
    line2.set_ydata(ylst)

    fig.canvas.draw()
plt.ioff()
plt.show()
print("final weights: ",m,b)
print("final mse: ",loss_function(m, b, df))











