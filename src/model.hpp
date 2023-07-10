#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>
// #include <omp.h>

#include <yaml-cpp/yaml.h>
#include <boost/numeric/odeint.hpp>
#include <boost/config.hpp>

#include "pcg-cpp/include/pcg_random.hpp"

using namespace std;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// SOME USEFULL typedef
typedef vector<double>   state_type_iaf_cond_alpha;
typedef vector<double>   state_type_aeif_cond_exp;
typedef vector<double>   state_type_aqif_cond_exp;
typedef vector<double>   state_type_iaf_cond_exp;

#endif

// SOME USEFULL GLOBAL CONSTANT (if more than one consider using namespace)
/// External input with frequency below this threshold (expressed in kHz) will be ignored
const double   EPSILON = 0.0001;
/// Parameter setting the precision for the precision in the bisection procedure followed by Network::find_sol_bisection (used in case of oscillatory external poissonian input)
const double   TOLL = 0.0001;


// SOME UTILITY FUNCTIONS:
/// Templated function printing a vector
template <class T>
void print_vector(const vector<T> & v);

/// Function managing config file names
string join_and_correct_config(string conf, string dest);


/// Class useful to generate a (pseudo)random double between 0 and 1 using the <b>pcg</b> generator
class RandomGenerator {
private:
    pcg32                                              gen;
    uniform_real_distribution<double>                  dis;
public:
    RandomGenerator() {
        pcg32                                          _gen(pcg_extras::seed_seq_from<random_device>{});
        uniform_real_distribution<double>              _dis(0,1);
        gen = _gen;
        dis = _dis;
    }
    /// Function returning a (pseudo)random double between 0 and 1
    double getRandomUniform() {
        return  dis(gen);
    }
};

/// <hr>
/// Class handling the neuron structure
class Neuron {
public:
    /// State vector
    vector<double>                  x;
    /// Dimenssion of the state vector #x
    unsigned                        dim;
    /// time of last spike (initialized to -1 - #SubNetwork::t_ref) [ms]
    double                          t_last_spike;
    /// First outwards neighbors.
    /// A dictionary is implemetnted with format: {target SubNetwork : list of neurons}
    map<string, vector<unsigned>>   neighbors;
    /// Excitatory inputs [ms].
    /// A dictionary is implemetnted with format: {source SubNetwork : list of input times}
    map<string, vector<double>>     input_t_ex;
    /// Inhibitory inputs [ms].
    /// A dictionary is implemetnted with format: {source SubNetwork : list of input times}
    map<string, vector<double>>     input_t_in;

    /// Dictionary containing the index of the next relevant spike in the vector #input_t_ex for each excitatory Subnetwork
    map<string, unsigned>           next_sp_ex_index;
    /// Dictionary containing the index of the next relevant spike in the vector #input_t_in for each inhibitory Subnetwork
    map<string, unsigned>           next_sp_in_index;

    /// External input weight:
    /// for each neuron in the subnetwork, it is given by #SubNetwork::weights_ex['ext'] plus a random value uniformly extracted in [-#SubNetwork::dev_ext_weight, #SubNetwork::dev_ext_weight] [nS]
    double                          ext_weight;

    /// Vector of the neuron spike times [ms]
    vector<double>                  t_spikes;

    /// Class constructor
    Neuron(unsigned _dim);

    /// Method printing a list of the attributes and their value
    void info();


};


/// Class handling the SubNetwork structure
class SubNetwork {
private:
    /// RandomGenerator object useful to generate random numbers from uniform a uniform distribution in [0,1]
    RandomGenerator     g;
public:
    /// Name of the SubNetwork (must be unique in the Network)
    string       name;
   /*!  Model of neurons (Name-conventions of <a href="https://nest-simulator.readthedocs.io/en/v3.3/contents.html">NEST-simulator</a> adopted).
    *   Supported models:
    *     - iaf_cond_alpha (id_model 0)
    *     - aeif_cond_exp (id_model 1)
    *     - aqif_cond_exp (id_model 2)
    *     - aqif2_cond_exp (id_model 3)
    *     - iaf_cond_exp (id_model 4)
    *
    *  In order to add your own model you need to:  XXX complete here
    */
    string       neuron_model;
    /// Identificative number for the neuron model
    unsigned     id_model;
    /// Number of neurons in the SubNetwork
    unsigned     N;
    /// Membrain capacity [pF]
    double       C_m;
    /// Resting potential [mV]
    double       E_L;
    /// Excitatory reversal potential [mV]
    double       E_ex;
    /// Inhibitory reversal potential [mV]
    double       E_in;
    /// Reset potential [mV]
    double       V_res;
    /// Threshold potential [mV]
    double       V_th;
    /// Refractory time [ms]
    double       t_ref;
    /// External injected current [pA]
    double       I_e;
    /// Amplitude of the oscillatory part of the external injected current [pA]
    double       osc_amp;
    /// Angular frequency of the oscillatory part of the external injected current [kHz]
    double       osc_omega;
    /// Amplitude of the oscillatory part of the external input rate [kHz]
    double       osc_amp_poiss;
    /// Angular frequency of the oscillatory part of the external input rate [kHz]
    double       osc_omega_poiss;
    /// Deviation of #weights_ex['ext'] from its central value [nS]
    double       dev_ext_weight;
    /// Input rate from external source [kHz] (here simulating cortical input)
    double       ext_in_rate;
    /// characteristic time of excitatory synaptic inputs [ms]
    double       tau_syn_ex;
    /// characteristic time of inhibitory synaptic inputs [ms]
    double       tau_syn_in;

    /// Weights of excitatory input connections.
    /// A dictionary is implemetnted with format: {source SubNetwork : weight [nS]}.
    map<string, double> weights_ex;
    /// Weights of inhibitory input connections.
    /// A dictionary is implemetnted with format: {source SubNetwork : weight [NS]}; all weights are positive.
    map<string, double> weights_in;

    /// Synaptic delays from the subnet to the other subnets.
    /// A dictionary is implemetnted with format: {target SubNetwork : delay [ms]}; all weights are positive.
    map<string, double> out_delays;
    /// Connection probabilities of the SubNetwork with the other SubNetworks.
    /// A dictionary is implemetnted with format: {target SubNetwork : probability}.
    map<string, double> probabilities_out;
    /// Characterization of the effect of spike on target population
    /// A dictionary is implemetnted with format: {target SubNetwork : bool}. If true the effect of a spike on the target population is reverted (e.g. an excitatory input on a iaf_cond_exp neuron decreases g_ex)
    map<string, bool>   reverse_effect;

    // MODEL SPECIFIC ATTRIBUTES
    /// Subthreshold adaptation [nS] (only adaptive models)
    double       a_adaptive;
    /// Spike-triggered adaptation: step_height of adaptation variable after spike emission [pA] (only adaptive models)
    double       b_adaptive;
    /// Characteristic decay time of the adaptation variable (only adaptive models)
    double       tau_w_adaptive;
    /// Spike detection threshold (only aeif_cond_exp and aqif_cond_exp models)
    double       V_peak;

    /// Membrain leakage conductance [nS] (only aeif_cond_exp and iaf_cond_alpha)
    double       g_L;

    /// Slope factor of exponential rise (only aeif_cond_exp)
    double       delta_T_aeif_cond_exp;

    /// k parameter of Izhikevich adaptive model [pA/mV<SUP>2</SUP>]
    // C_m dV/dt = k(V-V_th)(V-E_L) + input currents - w + I_e; tau_w dw/dt = a (V-E_L) - w; w -> w+b  (only aqif_cond_exp and aqif2_cond_exp)
    double       k_aqif_cond_exp;

    /// V_b parameter of Izhikevich adaptive fast spiking interneurons model (only aqif2_cond_exp)
    double       V_b_aqif2_cond_exp;


    /// Vector of #N #Neuron-type objects
    vector<Neuron>     pop;

    /// Vector of neurons whose state you want to save
    vector<unsigned>   to_save;

    /// class constructor
    SubNetwork(string _name, string _neuron_model, int _N, double _C_m, double _E_L,  \
               double _E_ex,double _E_in, double _V_res, double _V_th,                \
               double _t_ref, double _I_e, double _osc_amp, double _osc_omega,        \
               double _dev_ext_weight, double _ext_in_rate,                           \
               double _osc_amp_poiss, double _osc_omega_poiss,                        \
               double _tau_syn_ex, double _tau_syn_in, RandomGenerator _g);

    /// Method which save Neuron::t_spikes of each Neuron in #pop
    /// \param out_file output file XXX scrivi come viene stampato
    void save_t_spikes(string out_file);

    /// Method printing a list of the attributes and their value
    void info();
};

/// Class handling the network structure
class Network {
private:
    /// End time of simulation [ms]
    double      t_end;
    /// Current time during simulation (starts at #t=0 and ends at #t_end)
    double      t = 0;
    /// Number of calls to evolve method with argument #t_end / #n_step
    unsigned    n_step = 0;
    /// Time resolution of the simulation [ms]
    double       dt;
    /// Input yaml-file with Network composition and features.
    string      subnets_config_yaml;
    /// Input yaml-file with connection weights (positive weights are excitatory, negative weights are inhibitory)
    string      weights_config_yaml;
    /// Input yaml-file with connectivity probabilities between subnetworks and corresponding delays.
    /// Note that each delay must immediately follow the related probability
    string      connections_config_yaml;
    /// Input yaml-file with list of neurons whose state you want to save at each step.
    /// You can leave this file empty if you don't want to save any nuron state.
    string      to_save_config_yaml;
    /// Output directory of the simulation
    string      out_dir;

    /// Dictionary with format {#SubNetwork::name : related index in SubNetwork::subnets}
    map<string, unsigned>   subnet_name_index;
    /// Dictionary with format {#SubNetwork::name : related SubNetwork::N}
    map<string, unsigned>   subnet_name_N;
    /// Vector of #SubNetwork-type objects
    vector<SubNetwork>      subnets;
    /// RandomGenerator object useful to generate random numbers from uniform uniform distribution in [0,1]
    RandomGenerator         g;

    /// External input mode:
    /// - <i>0</i> (base mode): each neuron receives an indipendent poisson signal with mean frequency = #SubNetwork::ext_in_rate and possibly with the osccillatory component
    /// - <i>2</i> (with_correlation mode): implementation of method A (ask for details, not compatible with oscillatory input)
    unsigned                input_mode;

    // EXTERNAL MODE SPECIFIC ATTRIBUTES
    /// Parameter regulating the input correlation of the striatum populations (only in input_mode=2)
    double                  rho_corr;
    /// Vector containing the subnets indices corresponding to the population with correlatated inputs (only in input_mode=2)
    vector<unsigned>        corr_pops;
    /// Map containg the last time generated by the exponential distribution for the (partially correlated) external input
    map<string, double>     corr_last_time;

public:

    /// Dictionary containing the relation between the supported Neuron models and the dimension of its state vector
    map<string, unsigned>   supported_models = { {"iaf_cond_alpha", 5}, {"aeif_cond_exp", 4}, {"aqif_cond_exp", 4}, {"aqif2_cond_exp", 4}, {"iaf_cond_exp", 3}};
    /// Dictionary with format { SubNetwork::neuron_model : SubNetwork::id_model }
    map<string, unsigned>   subnet_model_id = { {"iaf_cond_alpha", 0}, {"aeif_cond_exp", 1}, {"aqif_cond_exp", 2}, {"aqif2_cond_exp", 3}, {"iaf_cond_exp", 4}};

    /// Class constructor  // XXX handle errors in createSubnet
    Network(double _t_end, unsigned _n_step, double _dt, unsigned _input_mode, string _subnets_config_yaml, string _weights_config_yaml, \
        string _connections_config_yaml, string _to_save_config_yaml, string _out_dir, RandomGenerator _g, string _input_mode_config="");

    /// Method initializing the #subnets vector using configurations files:
    /// - #subnets_config_yaml for the neurons' features;
    /// - #weights_config_yaml for the synaptic weights;
    /// - #connections_config_yaml for the connections features: probabilities and delays.
    void createSubnets();
    /// Method initializing the vector SubNetwork::pop of each SubNetwork in #subnets
    /// Neurons are initialized with Neuron::x[0] = SubNetwork::E_L of the belonging SubNetwork
    void createPops();

    /// Method evolving the network for time _T
    void evolve(double _T);

    /// Method updating external input according to #input_mode
    void externalInputUpdate();

    /// Function whose solutions==0 needs to be find in case of oscillatory external input rate
    double input_func(double y_, double r0_, double A_, double omega_, double t0_, double t_);

    /// Function determining next spike time in case of oscillatory external input rate
    double find_sol_bisection (double y_, double r0_, double A_, double omega_, double t0_);

    /// Method freeing memory from past spikes and performing a control operation over the state vector of each neuron
    void free_past();

    /// Method printing:
    /// - main characteristics of the Network composition
    /// - the simulation control variables
    void info();

    #ifndef DOXYGEN_SHOULD_SKIP_THIS

    /// integrator used to evolve the neurons status
    boost::numeric::odeint::runge_kutta4<state_type_aqif_cond_exp>                         stepper_aqif2_cond_exp;
    function<void(const state_type_aqif_cond_exp &, state_type_aqif_cond_exp &, double)>   integrator_odeint_aqif2;

    boost::numeric::odeint::runge_kutta4<state_type_aqif_cond_exp>                         stepper_aqif_cond_exp;
    function<void(const state_type_aqif_cond_exp &, state_type_aqif_cond_exp &, double)>   integrator_odeint_aqif;

    boost::numeric::odeint::runge_kutta4<state_type_aeif_cond_exp>                         stepper_aeif_cond_exp;
    function<void(const state_type_aeif_cond_exp &, state_type_aeif_cond_exp &, double)>   integrator_odeint_aeif;

    boost::numeric::odeint::runge_kutta4<state_type_iaf_cond_alpha>                        stepper_iaf_cond_alpha;
    function<void(const state_type_iaf_cond_alpha &, state_type_iaf_cond_alpha &, double)> integrator_odeint_iaf;

    boost::numeric::odeint::runge_kutta4<state_type_iaf_cond_exp>                          stepper_iaf_cond_exp;
    function<void(const state_type_iaf_cond_exp &, state_type_iaf_cond_exp &, double)>     integrator_odeint_iaf2;

    #endif /* DOXYGEN_SHOULD_SKIP_THIS */
};


#endif
