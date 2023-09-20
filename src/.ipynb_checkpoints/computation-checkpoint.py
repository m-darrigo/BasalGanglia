import python_utils as utils
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import neurodsp
import re

def computepsd(Dd,function,subnets):
    
    new_Dd = Dd.replace('.','_')
    new_function = re.sub(r"[^0-9]", "", function)
    function_only = re.sub(r"[^a-zA-Z]", "", function)
    sigm_par = ''
    if (function == 'sigmoid5'):
        sigm_par = '0.0004'
    if (function == 'sigmoid6'):
        sigm_par = '0.0006'
    if (function == 'sigmoid7'):
        sigm_par = '0.001'
    if (function == 'sigmoid8'):
        sigm_par = '0.005'
    
    #check if periodogram folder exists and if not create one
    periodogram_path = './periodograms/' + new_Dd + '_' + subnets + '_' + function
    if not os.path.exists(periodogram_path):
        os.makedirs(periodogram_path)    
        
    #funzione Dd
    data1 = "./build/output/n1/" + function + "_0.00_0.0000_0.00_" + Dd + "_1.00_1.00_1.6/ext_rateD2.txt"
    data = np.loadtxt(data1).T
    #spikesim
    s = utils.SpikeSim("./build/output/n1/" + function + "_0.00_0.0000_0.00_" + Dd + "_1.00_1.00_1.6", 'new_sim_parallel.yaml', 0, np.max(data[0]))
    
    #s.info()

    s.histogram('all', res = 10., save_img = periodogram_path + "/activity" + new_function + ".png")

    #for (k,v) in (s.MeanActivity()).items():
    #    print(f'Mean activity of {k} \t {v[0]:.2f} kHz\t N_neurons = {v[1]} \t Activity per Neuron = {v[2]*1000:.4f} Hz')

    #s.welch_spectogram(subnets, res = 5.)
    
    
    #dopamine depletion periodogram
    s.periodogramdd(pop=subnets, data=data1, dd_par=sigm_par, res=1., N_parseg=500, save_img = periodogram_path + "/"+ subnets + new_function + "_periodogram.png")
    
    
    #pickle.dump(output, open("./output_"+ subnets +".pkl", "wb"))
    
    # Estrai i risultati del periodogramma #######################################################
    
    output=s.periodogram(pop=subnets, res=1., N_parseg=200, save_img = periodogram_path + "/periodogram.png")
    
    frequencies = output[0]
    times = output[1]
    power_spectrum = output[2]
    
    # Definisci l'intervallo di frequenze di interesse #######################################################

    # Calcola il valore medio del power spectrum per ogni segmento nell'intervallo di frequenze
    #mean_power_spectrum = np.mean(power_spectrum[:, indici_intervallo], axis=1)
    x = []
    for i in power_spectrum:
        y = np.sum(i)
        x.append(y)
        #len(i[:]) = 18
    
    z = []
    for i in power_spectrum:
        y = np.max(i)
        z.append(y)
        #len(i[:]) = 18
    
    integral = np.trapz(power_spectrum, x=frequencies, axis=0)
    
    
    plt.figure()
    plt.plot(np.arange(0, 5001), x)
    plt.xlabel('t [sec]')
    plt.ylabel('x')
    plt.show()
           
    
#--main
args = sys.argv
numberargs = args.__len__()
if (numberargs >= 2):
    Dd = args[1]
    function = ""
    subnets = "STN"
    if numberargs >= 3:
        function = args[2]
    if numberargs == 4:
        subnets = args[3]
else:
    print('error')
    sys.exit()
    

computepsd(Dd,function,subnets)
