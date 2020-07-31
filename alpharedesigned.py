#librerías
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

class Experiment:
    def __init__(self):
        print('Gap:')
        self.gap = input()
        print('Velocidad:')
        self.v = input()
        self.carpeta = '/home/marcos/Dropbox/TFG Marcos Moreno/simulaciones/files_' + str(self.gap)
        self.archivo = 'data' + str(self.v) + '_'
        self.files=[]
        self.files = sorted(glob.glob(self.carpeta + '/' + self.archivo + '*.*'))
        self.num_exps = len(self.files)

def media(vector):
    media = []
    len_vector = len(vector[0])
    num_exps = len(vector)
    for i in range(len_vector):
        suma = 0
        for exp in range(num_exps):
            suma = suma + vector[exp][i]
        media.append(suma/num_exps)
    return media

def log_vectores(vector1,vector2):
    logvector1, logvector2 = [], []
    m = len(vector1)
    for i in range(m):
        if vector1[i] != 0:
            logvector1.append(math.log(vector1[i],10))
            logvector2.append(math.log(vector2[i],10))
    return logvector1, logvector2

def plotGraph(X,Y,resol):
    distancia = list(np.where(np.array(X) <= resol)[0])[-1]
    plotcharLengths()
    plt.plot(X[0:distancia],Y[0:distancia])
    punto1 = np.asarray(plt.ginput(1,timeout=0))
    punto2 = np.asarray(plt.ginput(1,timeout=0))
    plt.close()
    return punto1,punto2

def charLengths(gap,velocity):
    global l1, l2
    l1=(5.9*float(gap))/math.sqrt(float(velocity))
    l2=(69*float(gap))/float(velocity)
    return l1,l2

def plotcharLengths():
    plt.axvline(x=math.log(1/l1,10), color='r', linestyle='-')
    plt.axvline(x=math.log(1/l2,10), color='b', linestyle='-')

def interpolacion(X,Y,punto1,punto2):
    # lo que hace es coger las X y las Y para los puntos que estén entre los seleccionados
    # pasa a array todo y luego a listas pq sino no furula
    xNew = list(np.array(X)[(np.array(X) >= punto1[0][0]) & (np.array(X) <= punto2[0][0])])
    yNew = list(np.array(Y)[(np.array(X) >= punto1[0][0]) & (np.array(X) <= punto2[0][0])])

    xInterpol = np.linspace(xNew[0],xNew[-1],1000)
    yInterpol2 = np.interp(xInterpol,xNew, yNew)
    return xInterpol, yInterpol2

def fourier(file):
    A = np.loadtxt(file,float,'#','\t')
    x = A[:,0]
    h = A[:,1]
    q = (1/(x[1]-x[0]))*(1/(x[-1]))*np.arange(x[0],x[-1]/2,x[1]-x[0])
    Y = np.fft.fft(h)
    S = Y*np.conj(Y)
    return q,S

def windowsize(file):
    A = np.loadtxt(file,float,'#','\t')
    Y = A[:,1]
    Lc,wc = [],[]
    len_y=len(Y)
    step=40000
    for i in range(1,len_y):
        YNEW = []
        wi = []
        k = len_y-i
        p=0
        while (p+k)<=len_y:
            YNEW.append(Y[p:p+k])
            p=step+p
        len_YNEW=len(YNEW)
        for s in range(len_YNEW):
            a=0
            h=(1/k)*sum(YNEW[s])
            for j in range(k):
                a=a+(YNEW[s][j]-h)**2
            wi.append(math.sqrt((1/k)*a))
        wc.append(sum(wi)/len_YNEW)
        Lc.append(k)
    len_wc=len(wc)
    wo=[]
    for u in range(len_wc):
        wo.append(math.sqrt(abs(wc[u]**2-min(wc)**2)))
    return Lc,wo


def regression_fourier(xInterpol,yInterpol,logX,logY,resol):
    xReg = np.asarray(xInterpol).reshape(-1, 1)
    yReg = np.asarray(yInterpol).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(xReg, yReg)
    yPred = lr.predict(xReg)

    pendiente = lr.coef_[0][0]
    print('La pendiente correspondiente a Fourier es:', pendiente)
    distancia = list(np.where(np.array(logX) <= resol)[0])[-1]
    plotcharLengths()
    plt.plot(logX[0:distancia],logY[0:distancia],xReg,yPred)
    plt.show()
    return xReg,yPred

def regression_window(X,Y,punto1,punto2):
    #falta interpol
    xNew = list(np.array(X)[(np.array(X) <= punto1[0][0]) & (np.array(X) >= punto2[0][0])])
    yNew = list(np.array(Y)[(np.array(X) <= punto1[0][0]) & (np.array(X) >= punto2[0][0])])

    xReg = np.asarray(xNew).reshape(-1, 1)
    yReg = np.asarray(yNew).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(xReg, yReg)
    yPred = lr.predict(xReg)

    pendiente = lr.coef_[0][0]
    print('La pendiente correspondiente a Window-size es:', pendiente)
    plt.plot(X, Y, xReg, yPred)
    plt.show()
