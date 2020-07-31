import alpharedesigned as a
import numpy as np

experimento = a.Experiment()

acum_S = []
for exp in range(experimento.num_exps):
    q,S = a.fourier(experimento.files[exp])
    acum_S.append(S)
Smedia = a.media(acum_S)
a.charLengths(experimento.gap,experimento.v)
logq,logS = a.log_vectores(q,Smedia)
punto1, punto2 = a.plotGraph(logq,logS,0)
qInterpol, SInterpol = a.interpolacion(logq,logS,punto1,punto2)
Regq, RegS = a.regression_fourier(qInterpol,SInterpol,logq, logS, 0)
np.savez('0.5_0.66.txt', logq, logS)
np.savez('0.5_0.66reg.txt', Regq, RegS)
