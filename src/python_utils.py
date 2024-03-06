# Import Python standard libraries
import glob
import re
import pickle

import numpy as np
from scipy import stats
from scipy import signal
import yaml


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

def burst_sequence(sequence):
    n = 2 # number of points needed to identify a burst
    # Initialize variables for counting and tracking elements
    counts = [] # Stores counts of consecutive 1s or 0s
    count = 1 # Current count of consecutive elements
    prev_element = sequence[0] # The previous element in the sequence

    # Determine the starting element for counting
    if prev_element == 0:
        start_count = 0 # Start counting from 0 if the first element is 0
    else:
        start_count = 1 # Start counting from 1 if the first element is 1

    # Loop through the sequence starting from the second element
    for element in sequence[1:]:
        if element == prev_element:
            count += 1 # Increment count if the current element matches the previous one
        else:
            counts.append(count) # Append count to the list and reset for a new sequence
            count = 1 # Reset count for the new element
            prev_element = element # Update previous element to the current

    counts.append(count) # Append the last count

    # Adjust the initial count if the first element was 0
    if start_count == 0:
        counts.insert(0, 0) # Insert a count of 0 at the beginning

    # Initialize variables for burst detection
    count = 0 # Count of bursts
    burst = False # Flag to indicate if currently in a burst
    index = 0 # Index to track position in counts
    blen = 0 # Length of the current burst
    length = [] # List to store lengths of each burst

    # Loop through the counts to identify bursts
    for element in counts[:]:
        if (index % 2) == 0: # Check for sequences of 1s
            if element >= n: # A burst starts if there are n or more consecutive 1s
                burst = True

        if (index % 2) == 1: # Check for sequences of 0s
            if element >= n-1: # End a burst if there are n or more consecutive 0s
                if burst is True:
                    burst = False # End the burst
                    count += 1 # Increment burst count
                    length.append(blen) # Add the burst length to the list
                    blen = 0 # Reset burst length

        index += 1 # Move to the next element
        if burst is True:
            blen += element # Add the length of 1s to the current burst length

    # Check for an ongoing burst at the end of the sequence
    if burst is True:
        count += 1 # Finalize the last burst
        length.append(blen) # Add its length to the list

    # Calculate the average burst length
    if count > 0:
        average = sum(length)/count # Calculate average length if there are bursts
    else: 
        average = 0 # Set average to 0 if there are no bursts

    # Return the burst count, lengths of each burst, and the average burst length
    print('number of burst, length of every burst, mean length:', end=' ')
    return count, length, average


class SpikeSim:
    '''Class loading and parsing files given by a simulation.
    The main attributes are the simulation parameters and results:

    * end_t: end time of simulation
    * dt: time resolution of the simulation
    * input_mode: external input mode:

        - 0 (base mode): each neuron receives an indipendent poisson signal 
        with mean frequency = SubNetwork::ext_in_rate
        - 2 (paper mode): the input to the striatal population is correlated (ask for details)
    * rho_corr_paper (only with input_mode 2)
    * data: dictionary with spike times corresponding to each population;
    data['pop'] is a list of np.arrays each containing the activity of a neuron
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
        self.t_end = -1
        self.dt = 0
        self.input_mode = 0
        self.rho_corr_paper = -1

        self.data = {}
        self.subnets = []
        self.omegas = {}

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
            if  len(paper_configfile.split('/')) !=1 :
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
        if self.input_mode != 0:
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

    def periodogram(self, pop='', res=1., N_parseg=1000, save_img=''):
        '''Method computing the periodogram resulting from the spiking activity of the passed subnetwork'''

        x,_ = np.histogram(np.concatenate(self.data[pop]), bins = int((self.t_end-self.t_start)/res))
        fs = 1/res*1000
        noverlap=int(N_parseg*0.5) #indica la frazione di dati che si vanno a sovrapporre nelle finestre che scorrono
        f, t,sxx = signal.spectrogram(x, fs, nfft= 2000, nperseg = N_parseg, noverlap=noverlap) #nfft va messo alto per fare venire bene lo spettrogramma però viene fatto molto velocemente se è None

        print(f'nparseg = {N_parseg}\tnoverlap={noverlap}')

        if save_img == 'show':
            plt.pcolormesh(t, f, sxx, shading='gouraud')
            plt.ylim(8, 26)
            plt.colorbar()
            plt.title(pop)
            plt.ylabel('f [Hz]')
            plt.xlabel('t [s]')
            plt.show()

        elif save_img == '':
            pass

        else:
            plt.savefig(save_img, dpi = 500, facecolor='white')
            plt.close()

        return [f, t, sxx]


    def periodogramdd(self, pop='', data='', dd_par=float('inf'), res=1., N_parseg=500, save_img=''):
        '''Method computing the periodogram resulting from the spiking activity of the passed subnetwork
        dopamine depletion condition is plotted
        
        :from data we get the dd function times and values
        :dd_par is the steepness parameter of the sigmoid
        doesn't work for pop=all
        '''

        x,_ = np.histogram(np.concatenate(self.data[pop]), bins = int((self.t_end-self.t_start)/res))
        fs = 1/res*1000
        noverlap=int(N_parseg*0.8) #indica la frazione di dati che si vanno a sovrapporre nelle finestre che scorrono
        f, t, sxx = signal.spectrogram(x, fs, nfft= 1000,nperseg = N_parseg, noverlap=noverlap) #nfft va messo alto per fare venire bene lo spettrogramma però viene fatto molto velocemente se è None

        print(f'nparseg = {N_parseg}\tnoverlap={noverlap}')

        # Crea il grafico combinato
        _, ax1 = plt.subplots()

        # Grafico del periodogramma
        pcm = ax1.pcolormesh(t, f, np.log(sxx), shading='gouraud')
        ax1.set_ylim(8, 26)
        plt.title(f'Sigmoid time constant = {1000/dd_par} [s]')
        plt.colorbar(pcm, ax=ax1, aspect=20).set_label('Log Power')
        ax1.set_ylabel('f [Hz]')
        ax1.set_xlabel('t [s]')

        # Grafico della funzione
        data[0] = data[0]/1000
        data[1] = data[1]/1.083
        ax2 = ax1.twinx()
        ax2.plot(data[0], data[1], 'r-')
        ax2.set_xlim(0.25, np.max(data[0]) - 0.5)
        ax2.set_ylabel('Dopamine depletion', color='r')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.show()

        if save_img == '':
            print('ciao')
        else:
            plt.savefig(save_img, dpi = 500, facecolor='white')
            plt.close()

        return [f, t, sxx]


    def threshold(self, pop='', N_parseg=500, save_img=''):
        '''Method computing the threshold value that defines the presence of a burst
        '''

        f, t, sxx =self.periodogram(pop=pop, res=1., N_parseg=N_parseg, save_img=save_img)

        # faccio somma nel tempo per trovare frequenza max #######################################################
        sum1 = np.sum(sxx, axis=1) #somma per tutte frequenze

        # Ottieni l'indice del massimo della funzione
        max_index = np.argmax(sum1)

        sxx_max_index = sxx[max_index, :]

        alpha = np.quantile(sxx_max_index,0.75)

        # con media #######################################################

        mask = (8 < f) & (f < 26)
        sxx_lim = sxx[mask,:]

        pow_t = []
        for i in range(len(t)):
            pow_t.append( np.mean(sxx_lim[:, i]) )

        pow_t = np.array(pow_t)
        alpha1 = np.quantile(pow_t,0.75)

        return alpha, alpha1


    def threshold_imgs(self, pop='', N_parseg=500, save_img=''):
        '''Method computing the threshold value that defines the presence of a burst with picture
        '''

        f, t, sxx =self.periodogram(pop=pop, res=1., N_parseg=N_parseg, save_img=save_img)

        # faccio somma nel tempo per trovare frequenza max #######################################################
        sum1 = np.sum(sxx, axis=1) #somma per tutte frequenze

        # Ottieni l'indice del massimo della funzione
        max_index = np.argmax(sum1)
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
        sxx_max_index = sxx[np.argmax(sum1), :]

        # Creare un grafico dei valori nel tempo
        plt.plot(t, sxx_max_index)
        plt.xlabel('Tempo')
        plt.ylabel('Valori')
        plt.title(f'Valori nel tempo per {max_index} Hz')
        plt.show()

        plt.hist(sxx_max_index)
        plt.show()

        alpha = np.quantile(sxx_max_index,0.75)

        return alpha



    def welch_spectrogram(self, pop='', nparseg=1000, show=True, res=1., save_img='', Ns={}):
        '''Method computing the spectrogram resulting from the spiking activity of the passed subnetwork using the Welch method'''

        if pop.lower() == 'all':

            l = len(self.subnets)
            cols = 1 if l==1 else 2
            rows = round(l/cols) if (l%2==0 or l==1) else int(l/cols)+1

            to_ret = {}
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
                        plt.xlabel('f [Hz]')
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
                    pop = input('welch_spectrogram: enter subnetwork: ')
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
                    plt.xlabel('f [Hz]')
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

                return f, {pop: pow_welch_spect}

    def activityDistribution(self, pop = '', save_img=''):
        '''Method computing the distribution of the number of spike of each neuron in the subnetworks'''

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

                break


    @staticmethod
    def crossCorr(x, y, l, rescale=True):
        '''Method computing the cross correlation between two vectors mediated over subvectors of len l
        Notes:
        * l must be even and less than len(x)/2
        * the two vector must be of the same lenght
        * if rescale is True (default) each subvector is zscored before calculating the cross correlation; otherways only the mean is subtracted to the data
        '''

        print('using decorator')

        if l%2 == 1:
            print('ERROR: not even l passed to cross_corr...')
            exit()
        if len(x) != len(y):
            print(f'ERROR: not same lenght arrays passed to cross_corr... {len(x)} vs {len(y)}')
            exit()
        m = int(len(x)/l)-1
        print(f'convolution calculated with {m} blocks of size {l}')
        c = np.zeros(l)
        for i in range(m):
            start = i*l+int(l/2)                    # start of x
            end = (i+1)*l+int(l/2)                  # end of x
            if rescale:
                tmp_y = stats.zscore(y[start-int(l/2):end+int(l/2)])
                tmp_x = stats.zscore(x[start-int(l/2):end+int(l/2)])
            else:
                tmp_y = y[start-int(l/2):end+int(l/2)] - y[start-int(l/2):end+int(l/2)].mean()
                tmp_x = x[start-int(l/2):end+int(l/2)] - x[start-int(l/2):end+int(l/2)].mean()
            tmp_x = tmp_x[int(l/2):l+int(l/2)]
            for k in range(l):
                c[k] += ( tmp_x * tmp_y[k:l+k] ).sum()

            # plt.plot(np.arange(-int(l/2), int(l/2), 1), np.array(c/(M*l/2)))
            # plt.plot(np.arange(len(tmp_x)), tmp_x)
            # plt.plot(np.arange(len(tmp_y)), tmp_y)
            # plt.show()
        return (np.arange(-int(l/2), int(l/2), 1), np.array(c/(m*l/2)))
