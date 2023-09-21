import numpy as np
from scipy import stats
from scipy import signal
from scipy.optimize import curve_fit
import glob
import yaml
import re
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()

def save_pkl(obj, path):
    '''Function saving an object as a pickle file.

    :param obj: python object (list, dictionary...) to be saved
    :type obj: generic

    :param path: path to the object to be saved
    :type path: string
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    '''Function loading an object from a pickle file.

    :param path: path to the object to be loaded
    :type path: string
    '''

    with open(path, 'rb') as f:
        return pickle.load(f)

def readSpikes(file):
    '''Function reading spike times in the format produced by the simulation

    :param file: path to the file with spike times
    :type file: string
    '''
    l = []
    with open(file) as in_file:
        for line in in_file.readlines():
            n_list = [float(i) for i in line.split()]
            
            if len(l) <= round(n_list[0]):
                l.extend( [ [] for i in range( (round(n_list[0]) - len(l) + 1) ) ] )
            l[round(n_list[0])].extend(n_list[1:])

    for i in range(len(l)):
        l[i] = np.array(l[i])

    return l

def burst_sequence(sequenza):
    #questa funzione inizia contando gli 1 e poi alterna
    conteggi = []
    count = 1
    prev_element = sequenza[0]
    
    #controlla se il primo elemento è 0 o 1
    if prev_element == 0:
        start_count = 0
    else:
        start_count = 1
    
    #itera su sequenza a partire dal secondo elemento
    for element in sequenza[1:]:
        if element == prev_element:
            count += 1
        else:
            conteggi.append(count)
            count = 1
            prev_element = element
    
    conteggi.append(count)
    
    #se il primo elemento è 0 inserisci conteggio iniziale come 0
    if start_count == 0:
        conteggi.insert(0, 0)
    
    count = 0
    burst = False
    index = 0
    blen = 0
    lunghezza = []
    
    for element in conteggi[:]:
        if (index % 2) == 0:
            if element >= 5:
                burst = True
        
        if (index % 2) == 1:
            if element >= 5:
                if burst == True:
                    burst = False
                    count += 1  
                    lunghezza.append(blen)
                    blen = 0
                    
        index += 1
        if burst == True:
            blen = blen + element
    else:
        if burst == True:
            count += 1
            lunghezza.append(blen)
            
    if count > 0:
        avarage = sum(lunghezza)/count
    else: avarage = 0
    
    print('numero di burst, lunghezza di ogni burst, lunghezza media:')
    return count, lunghezza, avarage

class SpikeSim:
    '''Class loading and parsing files given by a simulation.
    The main attributes are the simulation parameters and results:

    * end_t: end time of simulation
    * dt: time resolution of the simulation
    * input_mode: external input mode:

        - 0 (base mode): each neuron receives an indipendent poisson signal with mean frequency = SubNetwork::ext_in_rate
        - 2 (paper mode): the input to the striatal population is correlated (ask for details)
    * rho_corr_paper (only with input_mode 2)
    * data: dictionary with spike times corresponding to each population; data['pop'] is a list of np.arrays each containing the activity of a neuron
    * subnets: a list of the SubNetworks in the simulation
    '''

    def __init__(self, path, sim_fname, neglect_t, neglect_t_end=-1, config_fname=''):
        '''Class constructor:

        :param path: path to the directory with output simulation files and configuration files
        :type path: string

        :param sim_fname: name of the simulation configuration file (inside the directory matched by path)
        :type sim_fname: string

        :param neglect_t: time (in ms) to be neglected at the beginning of the simulation
        :type sim_fname: float

        :param neglect_t_end: time (in ms) to be neglected at the end of the simulation
        :type sim_fname: float

        :param config_fname: name of the subnets_config_yaml configuration file (inside the directory matched by path)
        :type sim_fname: string
        '''
        self.input_dir = path
        self.sim_filename = sim_fname

        self.t_start = neglect_t
        self.t_end = neglect_t_end
        self.dt = 0
        self.input_mode = 0
        self.rho_corr_paper = -1

        self.data = dict()
        self.subnets = []
        self.omegas = dict()

        self.getParameterValues()
        self.loadData()
        self.necglectTime()

        if config_fname!='':
            with open(self.input_dir + '/' +config_fname) as file:
                in_dict = yaml.load(file, Loader=yaml.FullLoader)
            for d in in_dict:
                self.omegas[d['name']] = d['osc_omega']



    def getParameterValues(self):
        '''Method initializing the simulation parameters'''

        with open(self.input_dir + '/' +self.sim_filename) as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)

        dict_t_end = in_dict['t_end']
        if self.t_end < 0.:
            self.t_end = dict_t_end
        elif self.t_end > dict_t_end:
            print(f'ERROR: t_end is too big: max = {dict_t_end}, passed = {self.t_end}')
            exit()
        self.dt = in_dict['dt']
        self.input_mode = in_dict['input_mode']

        if self.input_mode != 0:
            paper_configfile = in_dict['input_mode_config']
            if ( len(paper_configfile.split('/')) !=1 ):
                paper_configfile = paper_configfile.split('/')[-1]

            with open(self.input_dir + '/'+paper_configfile) as file_paper:
                paper_dict = yaml.load(file_paper, Loader=yaml.FullLoader)
            self.rho_corr_paper = paper_dict['rho_corr']

    def loadData(self):
        '''Method loading spike times for each SubNetwork'''

        subnets_files = glob.glob(self.input_dir + '/*_spikes.txt')
        for f in subnets_files:
            pop = re.split('/|_', f)[-2]
            self.data[pop] = readSpikes(f)
            self.subnets.append(pop)

        self.subnets = sorted(self.subnets)


    def necglectTime(self):
        '''Method removing the spikes occurring before t_start and after t_end (if > 0)'''

        for i in self.data:
            for j in range( len(self.data[i]) ):
                if self.t_end < 0.:
                    self.data[i][j] = self.data[i][j][self.data[i][j]>self.t_start] - self.t_start
                else:
                    self.data[i][j] = self.data[i][j][ np.logical_and( self.data[i][j]>self.t_start, self.data[i][j]<self.t_end ) ] - self.t_start

    def info(self):
        '''Method printing the simulation parameters'''

        print('Simulation data from: ' + self.input_dir)
        print('\t simulation config file: ' + self.sim_filename)
        print('\t subnets in the network: ', self.subnets)
        print(f'\t t_start = {self.t_start} ms')
        print(f'\t t_end = {self.t_end} ms')
        print(f'\t dt = {self.dt} ms')
        print(f'\t input_mode = {self.input_mode}')
        if (self.input_mode != 0):
            print(f'\t rho_corr = {self.rho_corr_paper}')

    def saveData(self, path):
        save_pkl(self.data, path)

    def histogram(self, pop = '', dd_par=float('inf'), res=1., save_img=''):
        '''Method showing or saving the spiking activity of a given subnet

        :param pop: desidered population; if 'all' is passed all population are showed.
        :type pop: string

        :param res: time width of each bin in the histogram
        :type pop: float

        :param save_img: path and name of the file to be saved
        :type save_img: string
        '''

        pop_passed = True
        if pop == '':
            pop_passed = False


        if pop.lower() == 'all':
            plt.figure()
            for i,p in enumerate(sorted(self.subnets)):
                print(p)
                l = len(self.subnets)
                cols = 1 if l==1 else 2
                rows = round(l/cols) if (l%2==0 or l==1) else int(l/cols)+1
                try:
                    plt.subplot(rows,cols, i+1)
                    plt.ylabel(p)
                    plt.xlabel('t [ms]')
                    plt.hist( np.concatenate(self.data[p]), bins=int((self.t_end - self.t_start)/res) )
                except Exception as e: print(e)
            plt.tight_layout()
            plt.subplots_adjust(wspace = 0.4, hspace = 0.5)
            plt.suptitle(f'Sigmoid time constant = {1000/dd_par} [s]', y = 1)
            plt.savefig(save_img, dpi=500, facecolor='white')
            plt.show()
            #plt.close()
        else:
            while True:

                if pop == '':
                    pop = input('histogram: enter subnetwork: ')
                    if pop.lower() == 'stop':
                        break
                    if not pop in self.subnets:
                        print(f'No subnet with name "{pop}", try again...')
                        pop = ''
                        continue

                plt.hist(np.concatenate(self.data[pop]), bins=int((self.t_end - self.t_start)/res))
                plt.title(f'{pop} - {self.rho_corr_paper}')

                if save_img == '':
                    plt.show()
                else:
                    plt.savefig(save_img, dpi = 500)
                    plt.close()

                if pop_passed: break
                else: pop = ''

    def MeanActivity(self):
        '''Method computing the mean spiking activity of the subnets'''
        MAct = []
        Ns = []
        MActPerN = []

        for s in self.subnets:
            if len(self.data[s]) >0:
                counts = len( np.concatenate(self.data[s]) )
            else: counts = 0
            MAct.append( counts/(self.t_end - self.t_start) )
            n = len(self.data[s])
            Ns.append(n)
            if n > 0:
                MActPerN.append(counts/n/(self.t_end - self.t_start))
            else:
                MActPerN.append(0)

        return {self.subnets[i] : [ MAct[i], Ns[i], MActPerN[i] ] for i in range(len(self.subnets))}

    def periodogram(self, pop='', res=1., N_parseg=500, save_img=''):
        '''Method computing the periodogram resulting from the (z-scored) spiking activity of the passed subnetwork'''

        pop_passed = True
        if pop == '':
            pop_passed = False


        if pop.lower() == 'all':
            plt.figure()
            for i,p in enumerate(sorted(self.subnets)):
                print(p)
                try:
                    plt.subplot(4,2, i+1)
                    x,_ = np.histogram(np.concatenate(self.data[p]), bins = int((self.t_end-self.t_start)/res))
                    x = stats.zscore(x)
                    fs = 1/res*1000
                    f, t, Sxx = signal.spectrogram(x, fs, nperseg = N_parseg, noverlap=int(N_parseg/5))
                    plt.pcolormesh(t, f, Sxx, shading='gouraud')
                    # plt.pcolormesh(t, f, Sxx, shading='auto')
                    plt.ylim(0, 120)
                    plt.colorbar()
                    plt.ylabel(f'f [Hz] {p}')
                    
                except Exception as e: print(e)
            # plt.tight_layout()
            # plt.savefig(self.input_dir+'/activity.png', dpi=500)
            plt.show()
            print(f'nparseg = {N_parseg}\tnoverlap={int(N_parseg/5)}')
        else:
            while True:
                if pop == '':
                    pop = input('histogram: enter subnetwork: ')
                    if pop.lower() == 'stop':
                        break
                    if not pop in self.subnets:
                        print(f'No subnet with name "{pop}", try again...')
                        pop = ''
                        continue

                x,_ = np.histogram(np.concatenate(self.data[pop]), bins = int((self.t_end-self.t_start)/res))
                x = stats.zscore(x)
                fs = 1/res*1000
                f, t, Sxx = signal.spectrogram(x, fs, nfft= 10000,nperseg = N_parseg, noverlap=int(N_parseg/5))
                plt.pcolormesh(t, f, Sxx, shading='gouraud')
                plt.ylim(10, 25)
                plt.colorbar()
                plt.title(pop)
                plt.ylabel(f'f [Hz]')
                plt.xlabel('t [s]')
                print(f'nparseg = {N_parseg}\tnoverlap={int(N_parseg/5)}')

                if save_img == '':
                    plt.show()
                else:
                    plt.savefig(save_img, dpi = 500, facecolor='white')
                    plt.close()

                if pop_passed: break
                else: pop = ''
        return [f, t, Sxx]
    
    def periodogramdd(self, pop='', data='', dd_par=float('inf'), res=1., N_parseg=500, save_img=''):
        '''Method computing the periodogram resulting from the (z-scored) spiking activity of the passed subnetwork
        dopamine depletion condition is plotted
        
        :from data we get the dd function times and values
        :dd_par is the steepness parameter of the sigmoid
        doesn't work for pop=all
        '''

        data = np.loadtxt(data).T
        pop_passed = True
        if pop == '':
            pop_passed = False


        if pop.lower() == 'all':
            plt.figure()
            for i,p in enumerate(sorted(self.subnets)):
                print(p)
                try:
                    plt.subplot(4,2, i+1)
                    x,_ = np.histogram(np.concatenate(self.data[p]), bins = int((self.t_end-self.t_start)/res))
                    x = stats.zscore(x)
                    fs = 1/res*1000
                    f, t, Sxx = signal.spectrogram(x, fs, nperseg = N_parseg, noverlap=int(N_parseg/5))
                    plt.pcolormesh(t, f, Sxx, shading='gouraud')
                    # plt.pcolormesh(t, f, Sxx, shading='auto')
                    plt.ylim(0, 120)
                    plt.colorbar()
                    plt.ylabel(f'f [Hz] {p}')
                    
                except Exception as e: print(e)
            # plt.tight_layout()
            # plt.savefig(self.input_dir+'/activity.png', dpi=500)
            plt.show()
            print(f'nparseg = {N_parseg}\tnoverlap={int(N_parseg/5)}')
        else:
            while True:
                if pop == '':
                    pop = input('histogram: enter subnetwork: ')
                    if pop.lower() == 'stop':
                        break
                    if not pop in self.subnets:
                        print(f'No subnet with name "{pop}", try again...')
                        pop = ''
                        continue

                x,_ = np.histogram(np.concatenate(self.data[pop]), bins = int((self.t_end-self.t_start)/res))
                x = stats.zscore(x)
                fs = 1/res*1000
                f, t, Sxx = signal.spectrogram(x, fs, nfft= 10000,nperseg = N_parseg, noverlap=int(N_parseg/5))
                
                print(f'nparseg = {N_parseg}\tnoverlap={int(N_parseg/5)}')
         
                # Crea il grafico combinato
                fig, ax1 = plt.subplots()

                # Grafico del periodogramma
                pcm = ax1.pcolormesh(t, f, np.log(Sxx), shading='gouraud')
                ax1.set_ylim(10, 25)
                plt.title(f'Sigmoid time constant = {1000/dd_par} [s]')
                cbar = plt.colorbar(pcm, ax=ax1, location='left', aspect=20) 
                ax1.set_ylabel(f'f [Hz]')
                ax1.set_xlabel('t [s]')
                cbar.set_label('Log Power')
    
                # Grafico della funzione
                data[0] = data[0]/1000
                data[1] = data[1]/1.083
                ax2 = ax1.twinx()
                ax2.plot(data[0], data[1], 'r-')
                ax2.set_xlim(0.25, np.max(data[0]) - 0.5)
                ax2.set_ylabel('Dopamine depletion', color='r')
                ax2.tick_params(axis='y', labelcolor='red')

                if save_img == '':
                    plt.show()
                else:
                    plt.savefig(save_img, dpi = 500, facecolor='white')
                    plt.close()

                if pop_passed: break
                else: pop = ''
        return [f, t, Sxx]
    
    
    def threshold(self, pop='', data='', dd_par=float('inf'), res=1., N_parseg=500, save_img=''):
       
        output=self.periodogram(pop=pop, res=1., N_parseg=N_parseg, save_img=save_img)
        f = output[0]
        t = output[1]
        Sxx = output[2]
        
        # faccio somma nel tempo per trovare frequenza max #######################################################
        sum1 = np.sum(Sxx, axis=1) #somma per tutte frequenze
        
        # Ottieni l'indice del massimo della funzione
        max_index = np.argmax(sum1)/10
        max_value = sum1[np.argmax(sum1)]
        
        Sxx_max_index = Sxx[np.argmax(sum1), :]
        
        tau = np.quantile(Sxx_max_index,0.75)
        
        # con integrale #######################################################
        
        mask = (8 < f) & (f < 24)
        Sxx_lim = Sxx[mask,:]
        
        pow_t = []
        for i in range(len(t)):
            pow_t.append( np.mean(Sxx_lim[:, i]) )

        pow_t = np.array(pow_t)
        tau1 = np.quantile(pow_t,0.75)

        return tau, tau1
        
        
    def threshold_imgs(self, pop='', data='', dd_par=float('inf'), res=1., N_parseg=500, save_img=''):
       
        output=self.periodogram(pop=pop, res=1., N_parseg=N_parseg, save_img=save_img)
        f = output[0]
        t = output[1]
        Sxx = output[2]
        
        # faccio somma nel tempo per trovare frequenza max #######################################################
        sum1 = np.sum(Sxx, axis=1) #somma per tutte frequenze
        
        # Ottieni l'indice del massimo della funzione
        max_index = np.argmax(sum1)/10
        max_value = sum1[np.argmax(sum1)]
        print('Indice del massimo:', max_index)
        print('valore del massimo:', max_value)
        
        
        plt.figure()
        plt.plot(f, sum1)
        plt.xlabel('f [Hz]')
        plt.xlim(10, 25)
        plt.ylabel('x')
        plt.yscale('log')

        # Traccia una riga verticale sul massimo
        plt.axvline(x=max_index, color='red', linestyle='--')
        # Aggiungi un punto sul massimo
        plt.scatter(max_index, max_value, color='blue', marker='o')
        plt.show()
        
        # 2Estrarre la colonna corrispondente all'indice max_index
        Sxx_max_index = Sxx[np.argmax(sum1), :]

        # Creare un grafico dei valori nel tempo
        plt.plot(t, Sxx_max_index)
        plt.xlabel('Tempo')
        plt.ylabel('Valori')
        plt.title(f'Valori nel tempo per {max_index} Hz')
        plt.show()
        
        plt.hist(Sxx_max_index)
        plt.show()
        
        tau = np.quantile(Sxx_max_index,0.75)
        
        return tau
        
        
        
        
    
        
    def welch_spectogram(self, pop='', nparseg=1000, show=True, res=1., save_img='', Ns={}):
        '''Method computing the spectrogram resulting from the spiking activity of the passed subnetwork using the Welch method'''
        pop_passed = True
        if pop == '':
            pop_passed = False

        if pop.lower() == 'all':

            l = len(self.subnets)
            cols = 1 if l==1 else 2
            rows = round(l/cols) if (l%2==0 or l==1) else int(l/cols)+1

            to_ret = dict()
            if show:
                plt.figure()
            for i,p in enumerate(sorted(self.subnets)):
                # print(p)
                try:
                    x,_ = np.histogram(np.concatenate(self.data[p]), bins = int((self.t_end-self.t_start)/res))
                    # x = stats.zscore(x)
                    print('not_z_scored')
                    fs = 1/res*1000
                    if Ns != {}:
                        print('do not zsc!')
                        f, pow_welch_spect = signal.welch(x/Ns[p], fs, nperseg=nparseg, noverlap=int(nparseg/5),nfft=max(30000,nparseg), scaling='density', window='hamming')
                    else:
                        f, pow_welch_spect = signal.welch(x, fs, nperseg=nparseg, noverlap=int(nparseg/5),nfft=max(30000,nparseg), scaling='density', window='hamming')
                    if show:
                        plt.subplot(rows,cols, i+1)
                        plt.plot(f, pow_welch_spect)
                        plt.xlabel(f'f [Hz]')
                        plt.ylabel(f'PSD {p} [u.a.]')
                        plt.xlim(0, 120)
                        # plt.yscale('log')

                    to_ret[p] = pow_welch_spect

                except Exception as e: print(e)
            # plt.tight_layout()
            # plt.savefig(self.input_dir+'/activity.png', dpi=500)
            if show:
                plt.show()
            return (f, to_ret)
        else:
            while True:
                if pop == '':
                    pop = input('welch_spectogram: enter subnetwork: ')
                    if pop.lower() == 'stop':
                        break
                    if not pop in self.subnets:
                        print(f'No subnet with name "{pop}", try again...')
                        pop = ''
                        continue

                x,_ = np.histogram(np.concatenate(self.data[pop]), bins = int((self.t_end-self.t_start)/res))

                # x = stats.zscore(x)
                print('not_z_scored')
                fs = 1/res*1000
                f, pow_welch_spect = signal.welch(x, fs, nperseg=nparseg, noverlap=int(nparseg/5), nfft=max(30000,nparseg), scaling='density')
                if show or save_img!='' :
                    plt.plot(f, pow_welch_spect)
                    plt.xlabel(f'f [Hz]')
                    plt.ylabel(f'PSD {pop}')
                    # plt.ylim(0, 120)
                    # plt.yscale('log')

                    np.random.shuffle(x)
                    f, pow_welch_spect = signal.welch(x, fs, nperseg=nparseg, noverlap=int(nparseg/5), scaling='density')
                    plt.plot(f, pow_welch_spect, label='shuffled', color='black', linewidth=0.7)
                    plt.legend()
                    # plt.plot([min(f),max(f)], [0.001]*2)
                    # plt.xlim(0, 300)

                if save_img == '':
                    if show:
                        plt.show()
                else:
                    plt.savefig(save_img, dpi = 500)
                    plt.close()

                if pop_passed: return f, {pop: pow_welch_spect}
                else: pop = ''

    def activityDistribution(self, pop = '', save_img=''):
        '''Method computing the distribution of the number of spike of each neuron in the subnetworks'''

        pop_passed = True
        if pop == '':
            pop_passed = False


        if pop.lower() == 'all':
            plt.figure()
            for i,p in enumerate(sorted(self.subnets)):
                print(p)
                try:
                    plt.subplot(4,2, i+1)
                    plt.ylabel(p)
                    tmp = [ len(l) for l in self.data[p] ]
                    plt.hist( tmp, bins=range(min(tmp), max(tmp) + 1, 1) )
                except Exception as e: print(e)
            # plt.tight_layout()
            # plt.savefig(self.input_dir+'/activity.png', dpi=500)
            plt.show()
        else:
            while True:

                if pop == '':
                    pop = input('activity distribution: enter subnetwork: ')
                    if pop.lower() == 'stop':
                        break
                    if not pop in self.subnets:
                        print(f'No subnet with name "{pop}", try again...')
                        pop = ''
                        continue

                plt.hist([ len(l) for l in self.data[pop] ])
                plt.title(f'{pop} - {self.rho_corr_paper}')

                if save_img == '':
                    plt.show()
                else:
                    plt.savefig(save_img, dpi = 500)
                    plt.close()

                if pop_passed: break
                else: pop = ''

    @staticmethod
    def crossCorr(x, y, L, rescale=True):
        '''Method computing the cross correlation between two vectors mediated over subvectors of len L
        Notes:
        * L must be even and less than len(x)/2
        * the two vector must be of the same lenght
        * if rescale is True (default) each subvector is zscored before calculating the cross correlation; otherways only the mean is subtracted to the data
        '''

        print('using decorator')

        if L%2 == 1:
            print('ERROR: not even L passed to cross_corr...')
            exit()
        if len(x) != len(y):
            print(f'ERROR: not same lenght arrays passed to cross_corr... {len(x)} vs {len(y)}')
            exit()
        M = int(len(x)/L)-1
        print(f'convolution calculated with {M} blocks of size {L}')
        c = np.zeros(L)
        for i in range(M):
            start = i*L+int(L/2)                    # start of x
            end = (i+1)*L+int(L/2)                  # end of x
            if rescale:
                tmp_y = stats.zscore(y[start-int(L/2):end+int(L/2)])
                tmp_x = stats.zscore(x[start-int(L/2):end+int(L/2)])
            else:
                tmp_y = y[start-int(L/2):end+int(L/2)] - y[start-int(L/2):end+int(L/2)].mean()
                tmp_x = x[start-int(L/2):end+int(L/2)] - x[start-int(L/2):end+int(L/2)].mean()
            tmp_x = tmp_x[int(L/2):L+int(L/2)]
            for k in range(L):
                c[k] += ( tmp_x * tmp_y[k:L+k] ).sum()

            # plt.plot(np.arange(-int(L/2), int(L/2), 1), np.array(c/(M*L/2)))
            # plt.plot(np.arange(len(tmp_x)), tmp_x)
            # plt.plot(np.arange(len(tmp_y)), tmp_y)
            # plt.show()
        return (np.arange(-int(L/2), int(L/2), 1), np.array(c/(M*L/2)))

    def getAmp(self, x, L, res, omega):

        def f(t, A):
            return A*np.cos(omega*t*res)

        cc, corr = self.crossCorr(x,x,L, False)
        corr[np.argmax(corr)] = corr[np.argmax(corr[:int(L/2)-2])]

        pars,covm=curve_fit(f,cc,corr,[1])

        plt.plot(cc, corr)
        plt.plot(cc, f(cc, pars[0]))
        plt.title(self.input_dir)
        plt.show()

        return pars[0], np.sqrt(covm[0][0])
        

