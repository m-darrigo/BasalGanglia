#include <iostream>
#include "model.hpp"


using namespace std;

// SOME UTILITY FUNCTIONS:
template <class T>
void print_vector(const vector<T> & v) {
    if (v.size() > 0) {
        for (unsigned i=0; i<v.size()-1; i++) {
            cout << v[i] << " ";
        }
        cout << v[v.size()-1] << endl;
    }
    else cout << "empty" << endl;
}

string join_and_correct_config(string conf, string dest) {

    string      to_ret;
    to_ret = conf.substr( conf.find("/"), conf.length()-conf.find("/") );
    to_ret = dest +to_ret;
    return to_ret;
}

// Neuron
Neuron::Neuron(unsigned _dim) {

    x.resize(_dim, 0);
    dim = _dim;
}

void Neuron::info() {
    cout << "\tx\t\t";
    print_vector(x);
    cout << "\tneighbours"<<endl;
    for (auto i : neighbors) {
        cout << "\t    " << i.first << "\t  ";
        print_vector(i.second);
    }
    cout << "\texternal inp\t";
    print_vector(input_t_ex["ext"]);
}


// SubNetwork
SubNetwork::SubNetwork(string _name, string _neuron_model, int _N, double _C_m, double _E_L,       \
           double _E_ex, double _E_in, double _V_res, double _V_th,                                \
           double _t_ref, double _I_e, double _osc_amp, double _osc_omega, double _dev_ext_weight, \
           double _ext_in_rate, double _osc_amp_poiss, double _osc_omega_poiss,                    \
           double _tau_syn_ex, double _tau_syn_in, RandomGenerator _g) {
    name = _name;
    neuron_model = _neuron_model;
    N = _N;
    C_m = _C_m;
    E_L = _E_L;
    E_ex = _E_ex;
    E_in = _E_in;
    V_res = _V_res;
    V_th = _V_th;
    t_ref = _t_ref;
    I_e = _I_e;
    osc_amp = _osc_amp;
    osc_omega = _osc_omega;
    dev_ext_weight = _dev_ext_weight;
    ext_in_rate = _ext_in_rate;
    osc_amp_poiss = _osc_amp_poiss;
    osc_omega_poiss = _osc_omega_poiss;
    tau_syn_ex = _tau_syn_ex;
    tau_syn_in = _tau_syn_in;
    g = _g;

}


void SubNetwork::save_t_spikes (string out_file) {
    ofstream        of(out_file, ios::app);
    for (unsigned i=0; i<N; i++) {
        if ( pop[i].t_spikes.size() > 0 ) {
            of << i << "  ";
            for (auto s : pop[i].t_spikes) {
                of << s << "  ";
            }
            of << endl;
        }
    }
}


void SubNetwork::info() {
    cout << "Subnetwork " << name <<  endl;
    cout << "\tneuron_model\t" << neuron_model << endl;
    cout << "\tid_model\t" << id_model << endl;
    cout << "\tN\t\t" << N << endl;
    cout << "\tC_m\t\t" << C_m << endl;
    cout << "\tE_L\t\t" << E_L << endl;
    cout << "\tE_ex\t\t" << E_ex << endl;
    cout << "\tE_in\t\t" << E_in << endl;
    cout << "\tV_res\t\t" << V_res << endl;
    cout << "\tV_th\t\t" << V_th << endl;
    cout << "\tt_ref\t\t" << t_ref << endl;
    cout << "\tI_e\t\t" << I_e << endl;
    cout << "\text_in_rate\t" << ext_in_rate << endl;
    cout << "\ttau_syn_ex\t" << tau_syn_ex << endl;
    cout << "\ttau_syn_in\t" << tau_syn_in << endl;

    if (neuron_model == "iaf_cond_alpha") {
        cout << "\tg_L\t\t" << g_L << endl;
    }

    if (neuron_model == "aeif_cond_exp") {
        cout << "\tg_L\t\t" << g_L << endl;
        cout << "\ta_adaptive       " << a_adaptive << endl;
        cout << "\tb_adaptive       " << b_adaptive << endl;
        cout << "\ttau_w_adaptive   " << tau_w_adaptive << endl;
        cout << "\tdelta_T_aeif_cond_exp " << delta_T_aeif_cond_exp << endl;
        cout << "\tV_peak  " << V_peak << endl;
    }

    if (neuron_model == "aqif_cond_exp") {
        cout << "\tk_aqif_cond_exp\t" << k_aqif_cond_exp << endl;
    }

    cout << "\tdev_ext_weight\t" << dev_ext_weight << endl;
    cout << "\tweights_ex FROM\t";
    if (weights_ex.size()==0) cout << "empty" << endl;
    else cout << endl;
    for (auto i : weights_ex) {
        cout << "\t    " << i.first << "\t\t" << i.second << endl;
    }
    cout << "\tweights_in FROM\t";
    if (weights_in.size()==0) cout << "empty" << endl;
    else cout << endl;
    for (auto i : weights_in) {
        cout << "\t    " << i.first << "\t\t" << i.second << endl;
    }

    cout << "\tout_delays TO" << endl;
    for (auto i : out_delays) {
        cout << "\t    " << i.first << "\t\t" << i.second << endl;
    }

    cout << "\tconn prob TO" << endl;
    for (auto i : probabilities_out) {
        cout << "\t    " << i.first << "\t\t" << i.second << endl;
    }

    cout << "\tto_save\t\t\t";
    print_vector(to_save);

    // for (auto i : pop) {
    //     i.info();
    // }

}


// Network
Network::Network(double _t_end, unsigned  _n_step, double _dt, unsigned _input_mode, string _subnets_config_yaml, string _weights_config_yaml, \
    string _connections_config_yaml, string _to_save_config_yaml, string _out_dir, RandomGenerator _g, string _input_mode_config) {
    cout << "Network constructor called" << endl;

    t_end = _t_end;
    n_step = _n_step;
    dt = _dt;
    if (t_end/n_step < dt) {
        cout << "\tERROR: t_end/n_step < dt" <<endl;
        exit(1);
    }
    input_mode = _input_mode;

    subnets_config_yaml = _subnets_config_yaml;
    weights_config_yaml = _weights_config_yaml;
    connections_config_yaml = _connections_config_yaml;
    to_save_config_yaml = _to_save_config_yaml;
    out_dir = _out_dir;
    g = _g;

    createSubnets();
    if (input_mode == 2) {
        ifstream            f_in_corr(_input_mode_config);
        YAML::Node          n_corr = YAML::Load(f_in_corr);

        system(("cp " + _input_mode_config + " ./" + out_dir ).c_str());

        try {
            rho_corr = n_corr["rho_corr"].as<double>();
            for (auto j : n_corr["corr_pops"]) {
                corr_pops.push_back( subnet_name_index[j.as<string>()] );
            }
        } catch (...) {
            cout << "\tERROR occurred while parsing file _input_mode_config (i.e. " << _input_mode_config << ")" << endl;
            exit(1);
        }
    }

    createPops();
    info();

    while (t < t_end) {
        if (t + t_end/n_step > t_end) evolve(t_end -t);
        else {
            evolve(t_end/n_step);
            free_past();
        }
    }
}


void Network::createSubnets() {

    try {
        ifstream        f_in(subnets_config_yaml);
        YAML::Node      config = YAML::Load(f_in);
        unsigned        index=0;

        if (f_in.fail()) {
            cout << "ERROR: subnets_config_yaml (i.e. " << subnets_config_yaml << ") cannot be opened" << endl;
            exit(1);
        }

        for (auto i : config) {

            if (subnet_model_id.find(i["neuron_model"].as<string>()) == subnet_model_id.end()){
                cout << "\tERROR: neuron model not supported: " << i["neuron_model"].as<string>() << endl;
                exit(1);
            }

            subnet_name_index[i["name"].as<string>()] = index;
            subnet_name_N[i["name"].as<string>()] = i["N"].as<unsigned>();
            subnets.push_back( SubNetwork(i["name"].as<string>(), i["neuron_model"].as<string>(), i["N"].as<unsigned>(), \
                     i["C_m"].as<double>(), i["E_L"].as<double>(), i["E_ex"].as<double>(), i["E_in"].as<double>(),           \
                     i["V_res"].as<double>(), i["V_th"].as<double>(), i["t_ref"].as<double>(),                              \
                     i["I_e"].as<double>(), i["osc_amp"].as<double>(), i["osc_omega"].as<double>(),                         \
                     i["dev_ext_weight"].as<double>(), i["ext_in_rate"].as<double>(),                                       \
                     i["osc_amp_poiss"].as<double>(), i["osc_omega_poiss"].as<double>(),                                    \
                     i["tau_syn_ex"].as<double>(),i["tau_syn_in"].as<double>(), g) );
            subnets[index].id_model = subnet_model_id[i["neuron_model"].as<string>()];

            // particular attributes
            if (subnets[index].neuron_model == "aeif_cond_exp" || subnets[index].neuron_model == "aqif_cond_exp" || subnets[index].neuron_model == "aqif2_cond_exp") {
                subnets[index].a_adaptive = i["a_adaptive"].as<double>();
                subnets[index].b_adaptive = i["b_adaptive"].as<double>();
                subnets[index].tau_w_adaptive = i["tau_w_adaptive"].as<double>();
                subnets[index].V_peak = i["V_peak"].as<double>();
                if (subnets[index].neuron_model == "aeif_cond_exp") {
                    subnets[index].delta_T_aeif_cond_exp = i["delta_T_aeif_cond_exp"].as<double>();
                    subnets[index].g_L = i["g_L"].as<double>();
                }
                else if (subnets[index].neuron_model == "aqif_cond_exp") {
                    subnets[index].k_aqif_cond_exp = i["k_aqif_cond_exp"].as<double>();
                }
                else if (subnets[index].neuron_model == "aqif2_cond_exp") {
                    subnets[index].k_aqif_cond_exp = i["k_aqif_cond_exp"].as<double>();
                    subnets[index].V_b_aqif2_cond_exp = i["V_b_aqif2_cond_exp"].as<double>();
                }
            }
            if (subnets[index].neuron_model == "iaf_cond_alpha") {
                subnets[index].g_L = i["g_L"].as<double>();
            }
            if (subnets[index].neuron_model == "iaf_cond_exp") {
                subnets[index].g_L = i["g_L"].as<double>();
                subnets[index].V_peak = subnets[index].V_th;
            }

            index += 1;
        }
        f_in.close();
    } catch (...) {
        cerr << "\tERROR occurred while parsing file subnets_config_yaml (i.e. " << subnets_config_yaml << ")" << endl;
        exit(1);
    }

    // to_save
    try {
        ifstream            f_in_save(to_save_config_yaml);
        YAML::Node          to_save = YAML::Load(f_in_save);

        if (f_in_save.fail()) {
            cout << "ERROR: to_save_config_yaml (i.e. " << to_save_config_yaml << ") cannot be opened" << endl;
            exit(1);
        }

        for (auto i : to_save){
            if (i.Type() == YAML::NodeType::Map) {
                for (auto j : i) {
                    subnets[ subnet_name_index[j.first.as<string>()] ].to_save = j.second.as<vector<unsigned>>();
                }
            }
        }
        f_in_save.close();
    } catch (...) {
        cerr << "\tERROR occurred while parsing file to_save_config_yaml (i.e. " << to_save_config_yaml << ")" << endl;
        exit(1);
    }

    // weights
    try {
        ifstream        f_in_w(weights_config_yaml);
        YAML::Node      config_w = YAML::Load(f_in_w);
        double           temp;

        if (f_in_w.fail()) {
            cout << "ERROR: weights_config_yaml (i.e. " << weights_config_yaml << ") cannot be opened" << endl;
            exit(1);
        }

        for (auto k=subnets.begin(); k!=subnets.end(); k++) {
        // for subnet in subnets
            for (auto i : config_w) {
            // for node in weights_config_yaml
                if (k->name == i["name"].as<string>()) {
                    // check correspondence between subnet name and node name

                    for (auto j : i) {
                        if (j.first.as<string>() == "name") continue;
                        temp = j.second.as<double>();
                        if (temp < 0) {
                            k->weights_in[j.first.as<string>()] = -temp;
                        }
                        else {
                            k->weights_ex[j.first.as<string>()] = temp;
                        }
                    }

                    for (auto iter=next(i.begin(),1); iter!=i.end(); iter=next(iter,1)) {

                        temp = iter->second.as<double>();
                        if (temp < 0) {
                            k->weights_in[iter->first.as<string>()] = -temp;
                        }
                        else {
                            k->weights_ex[iter->first.as<string>()] = temp;
                        }
                    }

                    break;
                }
            }
        }
        f_in_w.close();
    } catch (...) {
        cerr << "\tERROR occurred while parsing file weights_config_yaml (i.e. " << weights_config_yaml << ")" << endl;
        exit(1);
    }

    // connections probabilities and delays
    try {
        ifstream        f_in_conn(connections_config_yaml);
        YAML::Node      config_conn = YAML::Load(f_in_conn);
        double          prob;

        if (f_in_conn.fail()) {
            cout << "ERROR: connections_config_yaml (i.e. " << connections_config_yaml << ") cannot be opened" << endl;
            exit(1);
        }

        for (auto k=subnets.begin(); k!=subnets.end(); k++) {
        // for subnet in subnets
            for (auto i : config_conn) {
            //for node in connections_config_yaml
                if (k->name == i["name"].as<string>()) {
                // check correspondence between subnet name and node name

                    for (auto iter=next(i.begin(),1); iter!=i.end(); iter=next(iter,2)) {
                    // jumping the "name" entry read the probability and the delay
                        prob = iter->second.as<double>();
                        if (prob < 0) {
                            k->probabilities_out[iter->first.as<string>()] = -prob;
                            k->reverse_effect[iter->first.as<string>()] = true;
                        }
                        else {
                            k->probabilities_out[iter->first.as<string>()] = prob;
                            k->reverse_effect[iter->first.as<string>()] = false;
                        }

                        k->out_delays[iter->first.as<string>()] = (next(iter,1))->second.as<double>();
                    }
                    break;
                }
            }
        }
        f_in_conn.close();
    } catch (...) {
        cerr << "\tERROR occurred while parsing file connections_config_yaml (i.e. " << connections_config_yaml << ")" << endl;
        exit(1);
    }

}


void Network::createPops() {
    string                      population;
    double                      probability, tmp_sum;
    vector<unsigned>            neig_temp;
    map <string, unsigned>      neig_count;

    // Neuron initialization
    for (auto k=subnets.begin(); k!=subnets.end(); k++) {
    // for subnet in subnets
        for (unsigned i=0; i<k->N; i++){
        // for each neuron in the population
            (k->pop).push_back( Neuron(supported_models[k->neuron_model]) );
            (k->pop)[i].x[0] = k->E_L;
            (k->pop)[i].t_last_spike = -1 - k->t_ref;
            if (g.getRandomUniform() > 0.5) (k->pop)[i].ext_weight = k->weights_ex["ext"] + k->dev_ext_weight * g.getRandomUniform();
            else (k->pop)[i].ext_weight = k->weights_ex["ext"] - k->dev_ext_weight * g.getRandomUniform();

            for (auto item : k->weights_ex) {
                (k->pop)[i].next_sp_ex_index[item.first] = 0;
            }
            for (auto item : k->weights_in) {
                (k->pop)[i].next_sp_in_index[item.first] = 0;
            }
        }
    }

    // These two parts must be kept separated!

    for (auto k=subnets.begin(); k!=subnets.end(); k++) {
    // for subnet in subnets

        neig_count.clear();
        for (auto j : k->probabilities_out) {
            neig_count[j.first] = 0;
        }

        for (unsigned i=0; i<k->N; i++){
        // for each neuron in the subnet population

            // (outwards) neighbors initialization
            for (auto j : k->probabilities_out) {
            //  for item in dictionary probabilities out
                neig_temp.clear();
                population = j.first;
                probability = j.second;
                if (probability < 0) {
                    cerr << "\tERROR: negative probability detected" << endl;
                    exit(1);
                }
                for(unsigned n=0; n<subnet_name_N[population]; n++){
                    if (g.getRandomUniform()<probability) {
                        neig_temp.push_back(n);
                    }
                }
                (k->pop)[i].neighbors[population] = neig_temp;

                neig_count[population] += neig_temp.size();
            }


            //  exernal input initialization for input_mode 0 (base mode)
            if (input_mode == 0 && k->ext_in_rate > EPSILON){
                tmp_sum = 0;
                while (tmp_sum < dt) {
                    tmp_sum += (- log(g.getRandomUniform()) ) / k->ext_in_rate;
                    (k->pop)[i].input_t_ex["ext"].push_back(tmp_sum);
                }
            }
        } // end for over k->pop

        cout << k->name << " connected to " << endl;
        for (auto j : neig_count) {
            cout << "\t" << j.first << " with " <<  j.second << " tot conn:\t" << double(j.second)/k->N  << " per source Neuron\t" << double(j.second) / subnet_name_N[j.first] << " per target neuron"<< endl;
        }

        if (input_mode == 2 && k->ext_in_rate > EPSILON){

            // non striatum pop
            if ( find( corr_pops.begin(), corr_pops.end(), k-subnets.begin() ) == corr_pops.end() ) {
                cout << "\t\t\t" << k->name << " IS NOT in corr_pops!" << endl;

                for (unsigned i=0; i<k->N; i++) {
                // for each neuron in the population
                    tmp_sum = 0.;
                    while (tmp_sum < dt) {
                        tmp_sum += (- log(g.getRandomUniform()) ) / k->ext_in_rate;
                        (k->pop)[i].input_t_ex["ext"].push_back(tmp_sum);
                    }
                }
            }

            // striatum pop
            else {
                cout << "\t\t\t" << k->name << " IS in corr_pops!" << endl;
                tmp_sum = 0;
                while (tmp_sum < dt) {
                    tmp_sum += (- log(g.getRandomUniform()) ) / k->ext_in_rate * rho_corr;
                    for (unsigned i=0; i<k->N; i++) {
                        if (g.getRandomUniform() < rho_corr) {
                            (k->pop)[i].input_t_ex["ext"].push_back(tmp_sum);
                        }
                    }
                }
                corr_last_time[k->name] = tmp_sum;
            }

        }

    } // end for over subnets
}

//double Network::input_func(double y_, double r0_, double A_, double omega_, double t0_, double t_) {
//    return t_ - t0_ + A_/omega_ * (cos(omega_*t0_) - cos(omega_*t_)) - y_/r0_;
//}

// PER SCEGLIERE LA FUNZIONE BISOGNA COMMENTARE LE ALTRE: QUI: A_ = t_mid,  omega_ = sigm_par, r0_ è 0.85, 0,1*1.083 è (0.95-0.85)*1.083 

double Network::input_func(double y_, double r0_, double A_, double omega_, double t0_, double t_) {

    // SIGMOID penso sia corretto questo
    // return (r0_ + 0.1*1.083)*(t_ - t0_) + 0.1*1.083/omega_ * log((exp(A_ * omega_) + exp(omega_*t0_))/((exp(A_ * omega_) + exp(omega_*t_)))) - y_;

    // REVERSESIGMOID
    // return (r0_ + 0.1*1.083)*(t_ - t0_) - 0.1*1.083/omega_ * log((exp(A_ * omega_) + exp(omega_*t0_))/((exp(A_ * omega_) + exp(omega_*t_)))) - y_;

    //SIGMOIDPULSE
    return (r0_ + 0.1*1.083)*(t_ - t0_) + 0.1*1.083/omega_ * log((exp(A_ * omega_) + exp(omega_*t0_))/((exp(A_ * omega_) + exp(omega_*t_)))) - 0.1*1.083/omega_ * log((exp((A_+10000) * omega_) + exp(omega_*t0_))/((exp((A_+10000) * omega_) + exp(omega_*t_)))) - y_;

    //FLAT
    //return (A_) * (t_ - t0_);
}


double Network::find_sol_bisection(double y_, double r0_, double A_, double omega_, double t0_) {
    double          tD = t0_, tU = t0_+1./r0_, val = 1e10, ftD, ftU, tbar;

    ftD = input_func(y_,r0_,A_,omega_,t0_,tD);
    ftU = input_func(y_,r0_,A_,omega_,t0_,tU);

    while ( ftD*ftU > 0 ) {
        tD = tU;
        tU += 1./r0_;
        ftD = ftU;
        ftU = input_func(y_,r0_,A_,omega_,t0_,tU);
    }
    while (abs(val) > TOLL) {
        tbar = (tD+tU)/2.;
        val = input_func(y_,r0_,A_,omega_,t0_,tbar);
        if (val * ftD > 0) {
            tD = tbar;
        }
        else {
            tU = tbar;
        }
    }
    return tbar;
}

void integrator_iaf_cond_alpha(const state_type_iaf_cond_alpha &x, state_type_iaf_cond_alpha &dxdt, const double t,          \
        double s_C_m, double s_g_L, double s_E_L, double s_E_ex, double s_E_in, double s_I_e, double s_osc_amp, double s_osc_omega, \
        double s_tau_syn_ex, double s_tau_syn_in) {
    dxdt[0] = 1/s_C_m * ( - s_g_L*(x[0]-s_E_L) - x[1]*(x[0]-s_E_ex) - x[3]*(x[0]-s_E_in) + s_I_e \
                          + s_osc_amp*sin(s_osc_omega*t) );                                             // membrain potential
    dxdt[1] = -x[1]/s_tau_syn_ex + x[2];                                                                // excitatory syn conductance
    dxdt[2] = -x[2]/s_tau_syn_ex;                                                                       // excitatory backup variable
    dxdt[3] = -x[3]/s_tau_syn_in + x[4];                                                                // inhibitory syn conductance
    dxdt[4] = -x[4]/s_tau_syn_in;                                                                       // inhibitory backup variable
}

void integrator_aeif_cond_exp(const state_type_aeif_cond_exp &x, state_type_aeif_cond_exp &dxdt, const double t,             \
        double s_C_m, double s_g_L, double s_E_L, double s_E_ex, double s_E_in, double s_I_e, double s_osc_amp, double s_osc_omega, \
        double s_tau_syn_ex, double s_tau_syn_in, double s_V_th, double s_a, double s_tau_w, double s_delta_T) {
    dxdt[0] = 1/s_C_m * ( - s_g_L*(x[0]-s_E_L) + s_g_L*s_delta_T*exp((x[0]-s_V_th)/s_delta_T) \
                          - x[1]*(x[0]-s_E_ex) - x[2]*(x[0]-s_E_in) - x[3] + s_I_e \
                          + s_osc_amp*sin(s_osc_omega*t) );                                             // membrain potential
    dxdt[1] = -x[1]/s_tau_syn_ex;                                                                       // excitatory syn conductance
    dxdt[2] = -x[2]/s_tau_syn_in;                                                                       // inhibitory syn conductance
    dxdt[3] = -x[3]/s_tau_w + s_a/s_tau_w * (x[0]-s_E_L);                                               // adaptation variable
}

void integrator_aqif_cond_exp(const state_type_aqif_cond_exp &x, state_type_aqif_cond_exp &dxdt, const double t,             \
        double s_C_m, double s_k, double s_E_L, double s_E_ex, double s_E_in, double s_I_e, double s_osc_amp, double s_osc_omega,   \
        double s_tau_syn_ex, double s_tau_syn_in, double s_V_th, double s_a, double s_tau_w) {
    dxdt[0] = 1/s_C_m * ( s_k*(x[0]-s_E_L)*(x[0]-s_V_th) - x[1]*(x[0]-s_E_ex) - x[2]*(x[0]-s_E_in) \
                          - x[3] + s_I_e + s_osc_amp*sin(s_osc_omega*t) );                              // membrain potential
    dxdt[1] = -x[1]/s_tau_syn_ex;                                                                       // excitatory syn conductance
    dxdt[2] = -x[2]/s_tau_syn_in;                                                                       // inhibitory syn conductance
    dxdt[3] = -x[3]/s_tau_w + s_a/s_tau_w * (x[0]-s_E_L);                                               // adaptation variable
}

void integrator_aqif2_cond_exp(const state_type_aqif_cond_exp &x, state_type_aqif_cond_exp &dxdt, const double t,             \
        double s_C_m, double s_k, double s_E_L, double s_E_ex, double s_E_in, double s_I_e, double s_osc_amp, double s_osc_omega,    \
        double s_tau_syn_ex, double s_tau_syn_in, double s_V_th, double s_a, double s_tau_w, double s_V_b) {
    dxdt[0] = 1/s_C_m * ( s_k*(x[0]-s_E_L)*(x[0]-s_V_th) - x[1]*(x[0]-s_E_ex) \
                          - x[2]*(x[0]-s_E_in) - x[3] + s_I_e + s_osc_amp*sin(s_osc_omega*t) );         // membrain potential
    dxdt[1] = -x[1]/s_tau_syn_ex;                                                                       // excitatory syn conductance
    dxdt[2] = -x[2]/s_tau_syn_in;                                                                       // inhibitory syn conductance
    if (x[0] < s_V_b)   dxdt[3] = -x[3]/s_tau_w + s_a/s_tau_w * pow((x[0]-s_V_b),3);
    else                dxdt[3] = -x[3]/s_tau_w;                                                        // adaptation variable
}

void integrator_iaf_cond_exp(const state_type_iaf_cond_exp &x, state_type_iaf_cond_exp &dxdt, const double t,                  \
        double s_C_m, double s_g_L, double s_E_L, double s_E_ex, double s_E_in, double s_I_e, double s_osc_amp, double s_osc_omega,   \
        double s_tau_syn_ex, double s_tau_syn_in) {
    dxdt[0] = 1/s_C_m * ( -s_g_L*(x[0]-s_E_L) - x[1]*(x[0]-s_E_ex) - x[2]*(x[0]-s_E_in) + s_I_e \
                          + s_osc_amp*sin(s_osc_omega*t) );                                             // membrain potential
    dxdt[1] = -x[1]/s_tau_syn_ex;                                                                       // excitatory syn conductance
    dxdt[2] = -x[2]/s_tau_syn_in;                                                                       // inhibitory syn conductance
}


void Network::evolve(double _T) {

    double           V_before, w_before;
    double           t0=t;
    unsigned        j;
    bool            save_flag = false;
    ofstream        t_save;
    unsigned        st_dim, sub_ind, ind0;          // ind0 is useful to keep trace of an index (must be always initialized to some constant)
    map<string, vector<ofstream>>    to_save_files;


    for (auto k : subnets) {
        if (k.to_save.size() > 0) {
            save_flag = true;
            break;
        }
    }
    if (save_flag) {

        if (t < dt) {
        // first initialization of save
            mkdir((out_dir + "/neuron_states").c_str(), 0755);
        }
        t_save.open((out_dir + "/neuron_states/t.txt").c_str(), ios::app);

        for (auto k : subnets) {
            if (k.to_save.size() > 0) {
                for (unsigned u : k.to_save) {
                    to_save_files[k.name].push_back( ofstream((out_dir+"/neuron_states/" + k.name + "_" + to_string(u)+".txt").c_str() , ios::app) );
                }
            }
        }
    }


    while (t<t0+_T){

        if ( fmod( t/t_end *100, 1) < dt/t_end * 100 ) {
            cout  << round( t/t_end *100 ) <<" %" << endl;
        }

        externalInputUpdate();

        for (auto k=subnets.begin(); k!=subnets.end(); k++) {
        // for subnet in subnets

            if (k->id_model == subnet_model_id["iaf_cond_alpha"]) {

                integrator_odeint_iaf = [ &k ] \
                                  ( const state_type_iaf_cond_alpha &x, state_type_iaf_cond_alpha &dxdt, const double t ) mutable { \
                                    return ::integrator_iaf_cond_alpha(x, dxdt, t, k->C_m, k->g_L, k->E_L, k->E_ex, k->E_in, k->I_e, k->osc_amp, k->osc_omega, k->tau_syn_ex, k->tau_syn_in);};

                for (unsigned i=0; i<k->N; i++){
                // for each neuron in the subnet population

                    // integration step execution
                    stepper_iaf_cond_alpha.do_step(integrator_odeint_iaf, (k->pop)[i].x, t, dt);

                    for (auto item : (k->pop)[i].input_t_ex) {
                        if ((k->pop)[i].next_sp_ex_index[item.first] == item.second.size()) continue;

                        if (t > item.second[(k->pop)[i].next_sp_ex_index[item.first]]) {
                            // zzz
                            cout << "simulation time t > next_spike_ex pulse...: " << k->name << " <-- " << item.first << "  " << t << " < " << item.second[(k->pop)[i].next_sp_ex_index[item.first]] << endl;
                            exit(1);
                        }

                        j=0;
                        ind0 = (k->pop)[i].next_sp_ex_index[item.first];
                        // in case of external input the weight is stored in Neuron::ext_weight
                        if (item.first == "ext") {
                            while ( ind0 + j < item.second.size() && t <= item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {
                                (k->pop)[i].x[2] += ((k->pop)[i].ext_weight) * M_E / (k->tau_syn_ex);
                                j += 1;
                            }
                        }
                        else {
                            while ( ind0 + j < item.second.size() && t <= item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {
                                // eps>1
                                if ( subnets[ subnet_name_index[item.first] ].reverse_effect[ k->name ] ) {
                                    (k->pop)[i].x[2] -= (k->weights_ex[item.first]) * M_E / (k->tau_syn_ex);
                                }
                                else {
                                    (k->pop)[i].x[2] += (k->weights_ex[item.first]) * M_E / (k->tau_syn_ex);
                                }
                                j += 1;
                            }
                        }

                        (k->pop)[i].next_sp_ex_index[item.first] += j;
                    } // SPECIFICO iaf_cond_alpha

                    // adjust external input case
                    if ((k->pop)[i].next_sp_ex_index["ext"] != 0) {
                        (k->pop)[i].input_t_ex["ext"].erase( (k->pop)[i].input_t_ex["ext"].begin(), \
                                        (k->pop)[i].input_t_ex["ext"].begin()+(k->pop)[i].next_sp_ex_index["ext"]);
                        (k->pop)[i].next_sp_ex_index["ext"] = 0;
                    } // GENERICO


                    for (auto item : (k->pop)[i].input_t_in) {

                        if ((k->pop)[i].next_sp_in_index[item.first] == item.second.size()) continue;

                        if (t > item.second[(k->pop)[i].next_sp_in_index[item.first]]) {
                            // zzz
                            cout << "simulation time t > next_spike_in pulse...: " << k->name << " <-- " << item.first << "  " << t << " < " << item.second[(k->pop)[i].next_sp_in_index[item.first]] << endl;
                            exit(1);
                        }

                        j=0;
                        ind0 = (k->pop)[i].next_sp_in_index[item.first];
                        while ( ind0 + j < item.second.size() && t <=  item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {   // zzz un solo <=
                            // eps>1
                            if ( subnets[ subnet_name_index[item.first] ].reverse_effect[ k->name ] ) {
                                (k->pop)[i].x[4] -= (k->weights_in[item.first]) * M_E / (k->tau_syn_in);
                            }
                            else {
                                (k->pop)[i].x[4] += (k->weights_in[item.first]) * M_E / (k->tau_syn_in);
                            }
                            j += 1;
                        }

                        (k->pop)[i].next_sp_in_index[item.first] += j;
                    } // SPECIFICO iaf_cond_alpha

                    if ( t-(k->pop)[i].t_last_spike < k->t_ref ) {
                        (k->pop)[i].x[0] = k->V_res;
                    } // GENERICO

                    if ( (k->pop)[i].x[0] > k->V_th ) {
                        (k->pop)[i].x[0] = k->V_res;
                        (k->pop)[i].t_last_spike = t;
                        (k->pop)[i].t_spikes.push_back(t);

                        // add input spike to neighbors with proper delay
                        for (auto item : (k->pop)[i].neighbors) {
                            // Excitatory
                            if ( subnets[ subnet_name_index[item.first] ].weights_ex.find( k->name ) != subnets[ subnet_name_index[item.first] ].weights_ex.end() ) {
                                for (auto neur_index : item.second){
                                    subnets[ subnet_name_index[item.first] ].pop[ neur_index ].input_t_ex[ k->name ].push_back(t + k->out_delays[item.first]);
                                }
                            }
                            // else {       ZZZ
                            else if (subnets[ subnet_name_index[item.first] ].weights_in.find( k->name ) != subnets[ subnet_name_index[item.first] ].weights_in.end()) {
                                for (auto neur_index : item.second){
                                    subnets[ subnet_name_index[item.first] ].pop[ neur_index ].input_t_in[ k->name ].push_back(t + k->out_delays[item.first]);
                                }
                            }
                            else {
                                cout << "Error: incoming spike is neither excitatory nor inhibitory" <<endl;
                                exit(1);
                            }
                        }
                    } //SPECIFICO iaf_cond_alpha

                } // end for over pop
            } // endif "iaf_cond_alpha"

            else if ( k->id_model == subnet_model_id["aqif_cond_exp"] || k->id_model == subnet_model_id["aqif2_cond_exp"] \
                      || k->id_model == subnet_model_id["aeif_cond_exp"] || k->id_model == subnet_model_id["iaf_cond_exp"] ) {

                if (k->id_model == subnet_model_id["aeif_cond_exp"]) {
                    integrator_odeint_aeif = [ &k ] \
                                      ( const state_type_iaf_cond_alpha &x, state_type_iaf_cond_alpha &dxdt, const double t ) mutable { \
                                        return ::integrator_aeif_cond_exp(x, dxdt, t, k->C_m, k->g_L, k->E_L, k->E_ex, k->E_in, k->I_e, k->osc_amp, k->osc_omega, k->tau_syn_ex, k->tau_syn_in, \
                                                                          k->V_th, k->a_adaptive, k->tau_w_adaptive, k->delta_T_aeif_cond_exp);};
                }
                else if (k->id_model == subnet_model_id["aqif_cond_exp"]) {
                    integrator_odeint_aqif = [ &k ] \
                                      ( const state_type_aqif_cond_exp &x, state_type_aqif_cond_exp &dxdt, const double t ) mutable { \
                                        return ::integrator_aqif_cond_exp(x, dxdt, t, k->C_m, k->k_aqif_cond_exp, k->E_L, k->E_ex, k->E_in, k->I_e, k->osc_amp, k->osc_omega, k->tau_syn_ex, k->tau_syn_in, \
                                                                          k->V_th, k->a_adaptive, k->tau_w_adaptive);};
                }
                else if (k->id_model == subnet_model_id["aqif2_cond_exp"]) {
                    integrator_odeint_aqif2 = [ &k ] \
                                      ( const state_type_aqif_cond_exp &x, state_type_aqif_cond_exp &dxdt, const double t ) mutable { \
                                        return ::integrator_aqif2_cond_exp(x, dxdt, t, k->C_m, k->k_aqif_cond_exp, k->E_L, k->E_ex, k->E_in, k->I_e, k->osc_amp, k->osc_omega, k->tau_syn_ex, k->tau_syn_in, \
                                                                          k->V_th, k->a_adaptive, k->tau_w_adaptive, k->V_b_aqif2_cond_exp);};
                }
                else if (k->id_model == subnet_model_id["iaf_cond_exp"]) {
                    integrator_odeint_iaf2 = [ &k ] \
                                      ( const state_type_iaf_cond_exp &x, state_type_iaf_cond_exp &dxdt, const double t ) mutable { \
                                        return ::integrator_iaf_cond_exp(x, dxdt, t, k->C_m, k->g_L, k->E_L, k->E_ex, k->E_in, k->I_e, k->osc_amp, k->osc_omega, k->tau_syn_ex, k->tau_syn_in);};
                }

                for (unsigned i=0; i<k->N; i++){

                    // integration step execution
                    V_before = (k->pop)[i].x[0];
                    w_before = (k->pop)[i].x[3];
                    if (k->id_model == subnet_model_id["aeif_cond_exp"]) {
                        stepper_aeif_cond_exp.do_step(integrator_odeint_aeif, (k->pop)[i].x, t, dt);
                    }
                    else if (k->id_model == subnet_model_id["aqif_cond_exp"]) {
                        stepper_aqif_cond_exp.do_step(integrator_odeint_aqif, (k->pop)[i].x, t, dt);
                    }
                    else if (k->id_model == subnet_model_id["aqif2_cond_exp"]) {
                        stepper_aqif2_cond_exp.do_step(integrator_odeint_aqif2, (k->pop)[i].x, t, dt);
                    }
                    else if (k->id_model == subnet_model_id["iaf_cond_exp"]) {
                        stepper_iaf_cond_exp.do_step(integrator_odeint_iaf2, (k->pop)[i].x, t, dt);
                    }

                    if (isnan((k->pop)[i].x[0]) || abs(V_before - (k->pop)[i].x[0]) > 150. ) {
                        if (k->id_model == 2 || k->id_model == 3 || k->id_model ==4) {
                            //cout << "\t Error: x[0] is nan OR single step increase of x[0] was bigger than 150 mV (Pleae decrease dt)" << endl; // zzz
                        }
                        (k->pop)[i].x[0] = k->V_peak+EPSILON;
                        (k->pop)[i].x[3] = w_before + dt/k->tau_w_adaptive * (k->a_adaptive * (k->V_peak-k->E_L) - w_before);
                    }

                    for (auto item : (k->pop)[i].input_t_ex) {
                        if ((k->pop)[i].next_sp_ex_index[item.first] == item.second.size()) continue;
                        // questo caso sopra forse inutile, ma non fa male

                        if (t > item.second[(k->pop)[i].next_sp_ex_index[item.first]]) {
                            // zzz
                            cout << "simulation time t > next_spike_ex pulse...: " << k->name << " <-- " << item.first << "  " << t << " < " << item.second[(k->pop)[i].next_sp_ex_index[item.first]] << endl;
                            exit(1);
                        }

                        j=0;
                        ind0 = (k->pop)[i].next_sp_ex_index[item.first];

                        if (item.first == "ext") {
                            while ( ind0 + j < item.second.size() && t <= item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {
                                (k->pop)[i].x[1] += (k->pop)[i].ext_weight;
                                j += 1;
                            }
                        }
                        else {
                            while ( ind0 + j < item.second.size() && t <= item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {
                                // eps>1
                                if ( subnets[ subnet_name_index[item.first] ].reverse_effect[ k->name ] ) {
                                    (k->pop)[i].x[1] -= k->weights_ex[item.first];
                                }
                                else {
                                    (k->pop)[i].x[1] += k->weights_ex[item.first];
                                }
                                j += 1;
                            }
                        }

                        (k->pop)[i].next_sp_ex_index[item.first] += j;
                    } // SPECIFICO aeif_cond_exp e aqif_cond_exp e iaf_cond_exp

                    // adjust external input case
                    if ((k->pop)[i].next_sp_ex_index["ext"] != 0) {
                        (k->pop)[i].input_t_ex["ext"].erase( (k->pop)[i].input_t_ex["ext"].begin(), \
                                        (k->pop)[i].input_t_ex["ext"].begin()+(k->pop)[i].next_sp_ex_index["ext"]);
                        (k->pop)[i].next_sp_ex_index["ext"] = 0;
                    } // GENERICO

                    for (auto item : (k->pop)[i].input_t_in) {
                        if ((k->pop)[i].next_sp_in_index[item.first] == item.second.size()) continue;

                        if (t > item.second[(k->pop)[i].next_sp_in_index[item.first]]) {
                            // zzz
                            cout << "simulation time t > next_spike_in pulse...: " << k->name << " <-- " << item.first << "  " << t << " < " << item.second[(k->pop)[i].next_sp_in_index[item.first]] << endl;
                            exit(1);
                        }

                        j=0;
                        ind0 = (k->pop)[i].next_sp_in_index[item.first];
                        while ( ind0 + j < item.second.size() && t <= item.second[ ind0 + j ] && item.second[ ind0 + j ] < t+dt ) {
                            // eps>1
                            if ( subnets[ subnet_name_index[item.first] ].reverse_effect[ k->name ] ) {
                                (k->pop)[i].x[2] -= k->weights_in[item.first];
                            }
                            else {
                                (k->pop)[i].x[2] += k->weights_in[item.first];
                            }
                            j += 1;
                        }

                        (k->pop)[i].next_sp_in_index[item.first] += j;
                    } // SPECIFICO aeif_cond_exp e aqif_cond_exp e iaf_cond_exp

                    if ( t-(k->pop)[i].t_last_spike < k->t_ref ) {
                        (k->pop)[i].x[0] = k->V_res;
                    } // GENERICO

                    if ( (k->pop)[i].x[0] > k->V_peak ) {
                        (k->pop)[i].x[0] = k->V_res;
                        (k->pop)[i].t_last_spike = t;
                        (k->pop)[i].t_spikes.push_back(t);

                        // update adaptation variable x[3], if necessary
                        if (not(k->id_model == subnet_model_id["iaf_cond_exp"])) (k->pop)[i].x[3] += k->b_adaptive;

                        // add input spike to neighbors with proper delay
                        for (auto item : (k->pop)[i].neighbors) {
                            // Excitatory
                            if ( subnets[ subnet_name_index[item.first] ].weights_ex.find( k->name ) != subnets[ subnet_name_index[item.first] ].weights_ex.end() ) {
                                for (auto neur_index : item.second){
                                    subnets[ subnet_name_index[item.first] ].pop[ neur_index ].input_t_ex[ k->name ].push_back(t + k->out_delays[item.first]);
                                }
                            }
                            // else {   ZZZ
                            else if (subnets[ subnet_name_index[item.first] ].weights_in.find( k->name ) != subnets[ subnet_name_index[item.first] ].weights_in.end()) {
                                for (auto neur_index : item.second){
                                    subnets[ subnet_name_index[item.first] ].pop[ neur_index ].input_t_in[ k->name ].push_back(t + k->out_delays[item.first]);
                                }
                            }
                            else {
                                cout << "Error: incoming spike is neither excitatory nor inhibitory" <<endl;
                                exit(1);
                            }
                        }
                    } //SPECIFICO aeif_cond_exp e aqif_cond_exp e iaf_cond_exp

                } // end for over pop
            } // endif "aeif_cond_exp" or "aqif_cond_exp"

        } // end for on subnets

        // update outfiles with current state
        // double     phi = M_PI;
        // if ( fmod(t*subnets[0].osc_omega, 2*M_PI)<phi+M_PI/48 && fmod(t*subnets[0].osc_omega, 2*M_PI)>phi-M_PI/48 ) {
        if (true) {
            if (save_flag) {
                t_save << t << endl;
                for (auto item_it = to_save_files.begin(); item_it!=to_save_files.end(); item_it++) {
                    sub_ind = subnet_name_index[item_it->first];
                    st_dim = subnets[ sub_ind ].pop[0].dim;
                    ind0 = 0;
                    for (auto f_it = item_it->second.begin(); f_it != item_it->second.end(); f_it++) {
                        for (unsigned s=0; s<st_dim; s++) {
                            (*f_it) << subnets[sub_ind].pop[ subnets[sub_ind].to_save[ind0] ].x[s] << "  ";
                        }
                        (*f_it) << endl;
                        ind0 += 1;
                    }
                }
            } // end if save_flag

        }     // end save only in phase phi

        t = t +dt;
    } // end while over t

    if (save_flag) {
        for (auto item_it = to_save_files.begin(); item_it!=to_save_files.end(); item_it++) {
            for (auto f_it = item_it->second.begin(); f_it != item_it->second.end(); f_it++) {
                (*f_it).close();
            }
        }
    }

    for(auto k : subnets) {
        k.save_t_spikes(out_dir+"/"+k.name+"_spikes.txt");
    }
}


double f_rectangular(double t, double v_up) {
    double v_down, t_start, t_end;
    v_down = 0.85;
    t_start = 1500;
    t_end = t_start + 1000;
    if (t < t_start || t > t_end) {
        return v_up;
    }
    else { return v_down;}
}

double f_step(double t, double v_up) {
    double v_down, t_start;
    t_start = 1500;
    v_down = 0.85;
    if (t < t_start) {
        return v_up;
    }
    else { return v_down;}
}

double f_alpha(double t, double v_up) {
    double v_down, t_start, a, b, c;
    t_start = 3000;
    a = 30*2.1;
    b = 40*1.2;
    c = 1;

    v_down = v_up - c * (exp(-(t - t_start)/a) - exp(-(t - t_start)/b));

    if (t < t_start) {
        return v_up;
    }
    else { return v_down;}
}

double f_sigmoid(double t, double v_up) {
    double steepness, v_down, t_mid;
    t_mid = 8000;
    steepness = 0.01;
    v_down = 0.85*1.083;
    
    return v_up + (v_down-v_up)* 1/( 1+ exp( -steepness*(t-t_mid) ) );
}

double f_reversesigmoid(double t, double v_up) {
    double steepness, v_down, t_mid;
    t_mid = 13000;
    steepness = 0.002;
    v_down = 0.85*1.083;
    
    return v_down + (v_up-v_down)* 1/( 1+ exp( -steepness*(t-t_mid) ) );
}

double f_sigmoidpulse(double t, double v_up) {
    double steepness, v_down, t_mid1, t_mid2;
    t_mid1 = 23000;
    t_mid2 = 53000;
    steepness = 0.0005;
    v_down = 0.85*1.083;
    
    return v_up + (v_down-v_up)* 1/( 1+ exp( -steepness*(t-t_mid1) ) ) + (v_up-v_down)* 1/( 1+ exp( -steepness*(t-t_mid2) ) );
}

double f_flat(double v_up) {
    
    return v_up;
}


void Network::externalInputUpdate() {

    double           tmp_sum, tmp_end, tbar, tmp_y, ext_rate;
    if (input_mode == 0) {        // base mode
        for (auto k=subnets.begin(); k!=subnets.end(); k++) {
            ext_rate = k->ext_in_rate;
            if (k->name == "D2") { 
                string shape = ""; //choose shape

                if (shape == "rectangular") {
                    ext_rate = f_rectangular(t, k->ext_in_rate);
                }

                if (shape == "step") {
                    ext_rate = f_step(t, k->ext_in_rate);
                }

                if (shape == "alpha") {
                    ext_rate = f_alpha(t, k->ext_in_rate);
                }

                if (shape == "reversesigmoid") {
                    ext_rate = f_reversesigmoid(t, k->ext_in_rate);
                }

                if (shape == "sigmoid") {
                    ext_rate = f_sigmoid(t, k->ext_in_rate);
                }

                if (shape == "sigmoidpulse") {
                    ext_rate = f_sigmoidpulse(t, k->ext_in_rate);
                }

                if (shape == "flat") {
                    ext_rate = f_flat(k->ext_in_rate);
                }

                if (shape == "") {}
                
                

                ofstream      f((out_dir+"/ext_rateD2.txt").c_str(), ios::app  );
                f << t << "\t" << ext_rate << endl; 
            }// choose D2
            
            if (ext_rate<EPSILON) continue;

            for (unsigned i=0; i<k->N; i++) {
            // for neuron in pop
                if ( *((k->pop)[i].input_t_ex["ext"].end()-1) < t+dt ) {
                    tmp_end = *((k->pop)[i].input_t_ex["ext"].end()-1);
                    tmp_sum = 0;
                    if (k->osc_amp_poiss < EPSILON) {
                        while (tmp_sum < dt) {
                            tmp_sum += (- log(g.getRandomUniform()) ) / ext_rate;
                            (k->pop)[i].input_t_ex["ext"].push_back(tmp_end + tmp_sum);
                        }
                    }
                    else {
                        //ofstream        of(out_dir+"/prova_sigm_inp.txt", ios::app); //commented
                        tbar = tmp_end;
                        while (tmp_sum < dt) {
                            tmp_y = - log(g.getRandomUniform());
                            tbar = find_sol_bisection( tmp_y, 0.85*1.083, k->osc_amp_poiss, k->osc_omega_poiss, tbar ); //0.85*1.083 era k->ext_in_rate
                            (k->pop)[i].input_t_ex["ext"].push_back(tbar);
                            tmp_sum = tbar - tmp_end;
                            //of << tbar << endl; //commented
                        }
                        //of.close(); //commented
                    }
                }
            } // end for over neurons
        } // end for over subnets
    } // end of input_mode == 0 (base mode)

    if (input_mode == 2) {        // with_correlation mode A;  oscillatory ext_in_rate not supported
        for (auto k=subnets.begin(); k!=subnets.end(); k++) {
        // for subnet in subnets

            if (k->ext_in_rate < EPSILON) continue;

            // non striatum pop
            if ( find( corr_pops.begin(), corr_pops.end(), k-subnets.begin() ) == corr_pops.end() ) {
                for (unsigned i=0; i<k->N; i++) {
                // for each neuron in the population
                    if ( *((k->pop)[i].input_t_ex["ext"].end()-1) < t+dt ) {
                        tmp_sum = 0.;
                        tmp_end = *((k->pop)[i].input_t_ex["ext"].end()-1);
                        while (tmp_sum < dt) {
                            tmp_sum += (- log(g.getRandomUniform()) ) / k->ext_in_rate;
                            (k->pop)[i].input_t_ex["ext"].push_back(tmp_end + tmp_sum);
                        }
                    }
                }
            }

            // striatum pop
            else {
                if (corr_last_time[k->name] < t+dt){
                    tmp_sum = 0;
                    tmp_end = corr_last_time[k->name];
                    while (tmp_sum < dt) {
                        tmp_sum += (- log(g.getRandomUniform()) ) / k->ext_in_rate * rho_corr;
                        for (unsigned i=0; i<k->N; i++) {
                            if (g.getRandomUniform() < rho_corr) {
                                (k->pop)[i].input_t_ex["ext"].push_back(tmp_end + tmp_sum);
                            }
                        }
                    }
                    corr_last_time[k->name] = tmp_end + tmp_sum;
                }
            }
        } // end for over subnets
    } // end of input_mode == 2 (with_correlation mode A)

}


void Network::free_past() {

    for (auto k=subnets.begin(); k!=subnets.end(); k++) {
    // for subnet in subnets
        for (unsigned i=0; i<k->N; i++){
        // for each neuron in the subnet population

            // perform a control operation on the values of the neuron neuron_states
            for (unsigned s=0; s<(k->pop)[i].dim; s++) {
                switch (fpclassify((k->pop)[i].x[s])) {
                    case FP_INFINITE:   {
                        cout << "\tERROR: infinite number met at t=" << t << "("<< k->name << ", neuron " << i<< ", st " << s <<  ")" << endl;
                        exit(1);
                    }
                    case FP_NAN:        {
                        cout << "\tERROR: NaN met at t=" << t << "("<< k->name << ", neuron " << i<< ", st " << s <<  ")" << endl;
                        exit(1);
                    }
                    case FP_SUBNORMAL:  {
                        (k->pop)[i].x[s] = 0.;
                    }
                }
            }

            // free t_spikes
            (k->pop)[i].t_spikes.clear();

            // free_past of input_t_ex vectors
            for (auto item_it=(k->pop)[i].input_t_ex.begin(); item_it!=(k->pop)[i].input_t_ex.end(); item_it++) {
                if ((k->pop)[i].next_sp_ex_index[item_it->first] != 0) {
                    // zzz cout << "\t\t" << item_it->first << " pass from: " << item_it->second.size() << " to: " ;
                    (item_it->second).erase( (item_it->second).begin(), (item_it->second).begin() + (k->pop)[i].next_sp_ex_index[item_it->first] );
                    // note: if next_sp_ex_index[item_it->first] == input_t_ex[item_it->first], this correctly returns the empty vector
                    (k->pop)[i].next_sp_ex_index[item_it->first] = 0;
                    // zzz cout << item_it->second.size() << endl;
                }
            }
            // free_past of input_t_in vectors
            for (auto item_it=(k->pop)[i].input_t_in.begin(); item_it!=(k->pop)[i].input_t_in.end(); item_it++) {
                if ((k->pop)[i].next_sp_in_index[item_it->first] != 0) {
                    (item_it->second).erase( (item_it->second).begin(), (item_it->second).begin() + (k->pop)[i].next_sp_in_index[item_it->first] );
                    // note: if next_sp_ex_index[item_it->first] == input_t_ex[item_it->first], this correctly returns the empty vector
                    (k->pop)[i].next_sp_in_index[item_it->first] = 0;
                }
            }
        } // end for over pop
    } // end for over subnets
}


void Network::info() {
    cout << "Network info:" << endl;

    cout << "\tt_end \t\t\t" << t_end << endl;
    cout << "\tn_step \t\t\t" << n_step << endl;
    cout << "\tt (resolut)\t\t" << dt << endl;
    cout << "\tinput mode\t\t" << input_mode << endl;
    if (input_mode == 2) {
        cout << "\t\trho_corr\t" << rho_corr << endl;
        cout << "\t\tcorr_pops\t";
        for (auto i : corr_pops) {
            cout << subnets[i].name << "  ";
        }
        cout << endl;
    }
    cout << "\tsubnets_config_yaml     " << subnets_config_yaml << endl;
    cout << "\tweights_config_yaml     " << weights_config_yaml << endl;
    cout << "\tconnections_config_yaml " << connections_config_yaml << endl;
    cout << "\tto_save_config_yaml     " << to_save_config_yaml << endl;
    cout << "\tout_dir                 " << out_dir << endl;

    cout << "\tsupported neuron models\t";
    for (auto i : subnet_model_id) cout << i.first << " [" << i.second << "]   ";
    cout << endl;

    cout << "\tsubnets ordering \n";
    for (auto i : subnet_name_index) cout << "\t\t[" << i.second << "] " << i.first << endl;

    cout << "subnet info " << endl;
    for (auto k : subnets) k.info();
}
