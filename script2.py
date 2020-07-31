import alpharedesigned as a

experimento = a.Experiment()
q,S = a.fourier(experimento.files[0])
logq,logS = a.log_vectores(q,S)
punto1, punto2 = a.plotGraph(logq,logS)
qInterpol, SInterpol = a.interpolacion(logq,logS,punto1,punto2)
a.regression_fourier(qInterpol,SInterpol,logq, logS, 0)
