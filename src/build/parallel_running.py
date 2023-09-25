import numpy as np
from subprocess import run
import threading, queue
import os
import time
from datetime import datetime
import glob

parallel_runs = 1 #6
iterations = 1 #1
end_time = 200000 #60500 #8500 # 60500 ms

''' usiamo config2 come riferimento gamma agisce solo sui pesi D2 -> GPTI -> FSN -> D2, gamma1 e gamma2 li lasciamo per ora '''
''' niente popolazione ausiliarie -> no epsilon ed eta '''

gamma1_s = [1.]                   # controls connectivity within loop1  (loopB con striatum)
gamma2_s = [1.]                   # controls connectivity within loop2  (loopA con stn)
gamma_s = [1.6]                   # controls intensity of loop1 whitout affecting external currents

par1 = [0.95]                     # input to D2


par2 = ["sigmoid11"]            # cambia questo per differenziare: step, rectangular, alpha

# corrente oscillante in pA
osc_amps = [0.] #pA
osc_freqs = [0.] #omega
osc_to = set([]) # set(['D2'])

# input poisson
osc_amps_poiss = [0.]
osc_freqs_poiss = [0.] # frequences in kHz!!
osc_to_poiss = set([])
# external_input_update in network class in model cpp, file out stream to save

save_dir_ = 'output/n1'
des = ''
n = 1 # 1, 2, 4, 8

#

config_file_names = ['new_sim_parallel.yaml', 'config1.yaml', 'config_weights1.yaml', 'config_connections1.yaml', 'config_to_save1.yaml']

q = queue.Queue()

for gamma1 in gamma1_s:
    for gamma2 in gamma2_s:
        for gamma in gamma_s:
            for osc_freq in osc_freqs:
                for osc_amp in osc_amps:
                    for osc_freq_p in osc_freqs_poiss:
                        for osc_amp_p in osc_amps_poiss:
                            for pp1 in par1:
                                for pp2 in par2:
                                    save_dir = save_dir_+f'/{pp2}_{osc_amp_p:.2f}_{osc_freq_p:.4f}_{osc_amp:.2f}_{pp1:.2f}_{gamma1:.2f}_{gamma2:.2f}_{gamma}'
                                    for it in range(iterations):
                                        if iterations == 1:
                                            save_name = save_dir+des
                                        else:
                                            if it == 0: os.mkdir(save_dir)
                                            save_name = f'{save_dir}/{it}{des}'


                                        config_files = []

                                        # simulation params
                                        config_files.append(f'''t_end:                    {end_time}  # ms
dt:                             {0.1}   # ms
n_step:                         200
out_dir:                        {save_name}
gamma_1:                        {gamma1}
gamma_2:                        {gamma2}
gamma:                          {gamma}
osc_amp:                        {osc_amp}
osc_freq:                       {osc_freq}
''')
                                        # config params
                                        config_files.append(f'''- name:                   D1
  neuron_model:           aqif_cond_exp
  N:                      {int(6000*n)}
  C_m:                    15.2
  E_L:                    -78.2
  E_ex:                   0.
  E_in:                   -74.
  V_res:                  -60.
  V_th:                   -29.7
  t_ref:                  0.
  I_e:                    0.
  osc_amp:                {osc_amp if 'D1' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            {1.6*0.7}                         # !!modified!!
  osc_amp_poiss:          {osc_amp_p if 'D1' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             12.
  tau_syn_in:             10.
  k_aqif_cond_exp:        1.
  a_adaptive:             -20
  b_adaptive:             66.9
  tau_w_adaptive:         100.
  V_peak:                 40.
  k_aqif_cond_exp:        1.

- name:                   D2
  neuron_model:           aqif_cond_exp
  N:                      {int(6000*n)}
  C_m:                    15.2
  k_aqif_cond_exp:        1.
  E_L:                    -80.
  E_ex:                   0.
  E_in:                   -74.
  V_res:                  -60.
  V_th:                   -29.7
  t_ref:                  0.
  I_e:                    {0. - (1-gamma1)*80}         # -36
  osc_amp:                {osc_amp if 'D2' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            {1.083*pp1}                        # !!par1!!
  osc_amp_poiss:          {osc_amp_p if 'D2' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             12.
  tau_syn_in:             10.
  a_adaptive:             -20
  b_adaptive:             91.
  tau_w_adaptive:         100.
  V_peak:                 40.
  k_aqif_cond_exp:        1.

- name:                   FSN
  neuron_model:           aqif2_cond_exp
  N:                      {int(420*n)}
  C_m:                    80.
  E_L:                    -80.
  E_ex:                   0.
  E_in:                   -74.
  V_res:                  -60.
  V_th:                   -50.
  t_ref:                  0.
  I_e:                    {0. - (1-gamma1)*100}       # -26
  osc_amp:                {osc_amp if 'FSN' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            {0.787*1.2}                        # !!modified!!
  osc_amp_poiss:          {osc_amp_p if 'FSN' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             12.
  tau_syn_in:             10.
  k_aqif_cond_exp:        1.
  a_adaptive:             0.025
  b_adaptive:             0
  tau_w_adaptive:         5.
  V_peak:                 25.
  V_b_aqif2_cond_exp:     -55

- name:                   GPTI
  neuron_model:           aeif_cond_exp
  N:                      {int(780*n)}
  C_m:                    40.
  E_L:                    -55.1
  E_ex:                   0.
  E_in:                   -65.
  V_res:                  -60.
  V_th:                   -54.7
  g_L:                    1.
  t_ref:                  0.0
  I_e:                    {12. - (1-gamma1)*44}       # -32
  osc_amp:                {osc_amp if 'GPTI' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            1.53
  osc_amp_poiss:          {osc_amp_p if 'GPTI' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             10.
  tau_syn_in:             5.5
  a_adaptive:             2.5
  b_adaptive:             70.
  tau_w_adaptive:         20.
  delta_T_aeif_cond_exp:  1.7
  V_peak:                 15.

- name:                   GPTA
  neuron_model:           aeif_cond_exp
  N:                      {int(264*n)}
  C_m:                    60.
  E_L:                    -55.1
  E_ex:                   0.
  E_in:                   -65.
  V_res:                  -60.
  V_th:                   -54.7
  g_L:                    1.
  t_ref:                  0.0
  I_e:                    1.
  osc_amp:                {osc_amp if 'GPTA' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            0.17
  osc_amp_poiss:          {osc_amp_p if 'GPTA' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             10.
  tau_syn_in:             5.5
  a_adaptive:             2.5
  b_adaptive:             105.
  tau_w_adaptive:         20.
  delta_T_aeif_cond_exp:  2.55
  V_peak:                 15.

- name:                   STN
  neuron_model:           aeif_cond_exp
  N:                      {int(408*n)}
  C_m:                    60.
  E_L:                    -80.2
  E_ex:                   0.
  E_in:                   -84.
  V_res:                  -70.
  V_th:                   -64.0
  g_L:                    10.
  t_ref:                  0.0
  I_e:                    {5 - (1-gamma2)*2.4}
  osc_amp:                {osc_amp if 'STN' in osc_to else 0.}
  osc_omega:              {osc_freq*2*np.pi}
  dev_ext_weight:         0.05
  ext_in_rate:            0.5 # 0.25 x2
  osc_amp_poiss:          {osc_amp_p if 'STN' in osc_to_poiss else 0.}
  osc_omega_poiss:        {osc_freq_p*2*np.pi}
  tau_syn_ex:             4.
  tau_syn_in:             8.
  a_adaptive:             0.
  b_adaptive:             0.05
  tau_w_adaptive:         333.
  delta_T_aeif_cond_exp:  16.2
  V_peak:                 15.

# manca GPi
''')
                                        # config weights
                                        config_files.append(f'''- name:         D1   # added
  D1:            -0.12
  D2:            -0.36
  FSN:           -6.6
  GPTA:          -0.35
  ext:           0.45   #0.46
 
- name:                   D2
  FSN:           {gamma1*(-3.0) * gamma}
  D2:            -0.2
  ext:           0.45
  D1:            -0.3                # added
  GPTA:          -0.61               # added

- name:         FSN
  GPTI:          {gamma1*(-1.0) * gamma}
  FSN:           -0.5
  ext:           0.5
  GPTA:          -1.85               # added

- name:         GPTI
  D2:            {gamma1*(-0.8) * gamma}
  STN:           {gamma2*(0.42)}
  GPTI:          -1.2
  ext:           0.25
  GPTA:          -1.2                # added

- name:         GPTA                 # added
  GPTA:          -0.35
  GPTI:          -0.35
  STN:           0.13
  ext:           0.15

- name:         STN
  GPTI:          {gamma2*(-0.08)}
  ext:           0.25
''')
                                        # config connections
                                        config_files.append(f'''- name:             D1
  D1:               {0.0607/n}   # 364/6000
  D1_delay:         1.7
  D2:               {0.014/n}    # 84
  D2_delay:         1.7
  # GPi:              0.0833    # 500
  # GPi_delay:        7.

- name:              D2
  D1:                 {0.06533/n}               # 392/6000              added
  D1_delay:           1.7
  D2:                 {0.084/n}                 # 504/6000
  D2_delay:           1.7
  GPTI:               {0.0833/n}                # 500/6000
  GPTI_delay:         7.

- name:              FSN
  D1:                 {0.0381/n}                # 16/420                added
  D1_delay:           1.7
  FSN:                {0.0238/n}                # 10/420
  FSN_delay:          1.
  D2:                 {0.0262/n}                # 11/420
  D2_delay:           1.7

- name:              GPTI
  GPTI:               {0.0321/n}                # 25/780
  GPTI_delay:         1.
  GPTA:               {0.0321/n}                # 25/780                added
  GPTA_delay:         1.
  FSN:                {0.0128/n}                # 10/780
  FSN_delay:          7.
  STN:                {0.0385/n}                # 30/780
  STN_delay:          1.

- name:             GPTA                                          # added
  D1:                 {0.0379/n}                # 10/264
  D1_delay:           7.
  D2:                 {0.0379/n}                # 10
  D2_delay:           7.
  FSN:                {0.0379/n}                # 10
  FSN_delay:          7.
  GPTA:               {0.0189/n}                # 5
  GPTA_delay:         1.
  GPTI:               {0.0189/n}                # 5
  GPTI_delay:         1.

- name:              STN
  GPTA:               {0.0735/n}                # 30/408                added
  GPTA_delay:         2.
  GPTI:               {0.0735/n}                # 30/408
  GPTI_delay:         2.
''')
                                        # config to-save
                                        # config_files.append('')
                                        config_files.append('''- STN: [ 0,  1] ''')
                                        q.put( [(gamma1, gamma2, gamma, osc_freq, osc_amp, osc_to, it), config_files] )

def worker():
    while True:
        pars, config_f = q.get()

        # print(f'{threading.get_native_id()} started')
        # time.sleep(5)
        # print(f'\t{threading.get_native_id()} ended')

        try:
            # time check
            f = open(f'TEMP-{threading.get_native_id()}/new_started.txt', 'r')
            line = f.readline()
            print(line)
            last_run = datetime.strptime(line, "%d-%b-%Y\t%H:%M:%S.%f\n")
            while ( (datetime.now()-last_run).total_seconds() < 5. ):
                f.close()
                time.sleep(5.)
                f = open(f'TEMP-{threading.get_native_id()}/new_started.txt', 'r')
                line = f.readline()
                print("\t", line)
                last_run = datetime.strptime(line, "%d-%b-%Y\t%H:%M:%S.%f\n")
            f.close()
        except FileNotFoundError:
            os.mkdir(f'TEMP-{threading.get_native_id()}')

        gamma1, gamma2, gamma, osc_freq, osc_amp, osc_to, it = pars
        with open(f'TEMP-{threading.get_native_id()}/new_started.txt', 'w') as f:
            f.write(datetime.now().strftime("%d-%b-%Y\t%H:%M:%S.%f"))
            f.write(f'\ngamma1 = {gamma1:.2f}, gamma2 = {gamma2:.2f}, gamma = {gamma:.3f} freq = {osc_freq:.4f}, amp = {osc_amp}, osc_to = {osc_to}, it = {it}')

        for i,file in enumerate(config_file_names):
            with open(f'TEMP-{threading.get_native_id()}/'+file, "w") as f:
                if i == 0:                  # appending config_paths to new_sim_parallel
                    config_f[i] = config_f[i] + f'''subnets_config_yaml:            TEMP-{threading.get_native_id()}/config1.yaml
weights_config_yaml:            TEMP-{threading.get_native_id()}/config_weights1.yaml
connections_config_yaml:        TEMP-{threading.get_native_id()}/config_connections1.yaml
to_save_config_yaml:            TEMP-{threading.get_native_id()}/config_to_save1.yaml
input_mode:                     0
input_mode_config:              TEMP-{threading.get_native_id()}/config_input_mode.yaml          # still not supported
'''

                f.write(config_f[i])

        print('========================')
        print(f'gamma1 = {gamma1:.2f}, gamma2 = {gamma2:.2f}, gamma = {gamma:.3f} freq = {osc_freq:.4f}, amp = {osc_amp}, osc_to = {osc_to}, it = {it}\n')
        print(config_f[0])
        print('========================')

        run(f'./main TEMP-{threading.get_native_id()}/new_sim_parallel.yaml', shell=True)
        time.sleep(2)

        q.task_done()


# check before starting that no TEMP-* dirs exists
config_dirs = glob.glob('TEMP-*')
if len(config_dirs) != 0:
    print('TEMP* dirs exist:', config_dirs)
    exit()

for i in range(parallel_runs):
    threading.Thread(target=worker, daemon=True).start()
    time.sleep(6.)

q.join()

config_dirs = glob.glob('TEMP-*')
if len(config_dirs) == parallel_runs:
    for d in config_dirs:
        os.system('rm -r {d}')
else:
    print('ERROR: config dirs not deleted')
