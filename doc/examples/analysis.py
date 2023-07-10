import python_utils as utils

s = utils.SpikeSim("./build/output/example_long", 'example.yaml', 500, 6500.)

s.info()

s.histogram('all', res = 10.)

for (k,v) in (s.MeanActivity()).items():
    print(f'Mean activity of {k} \t {v[0]:.2f} kHz\t N_neurons = {v[1]} \t Activity per Neuron = {v[2]*1000:.4f} Hz')

s.welch_spectogram('S', res = 5.)
