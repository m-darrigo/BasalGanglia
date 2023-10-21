#include <iostream>
#include "model.hpp"

#include <errno.h>

using namespace std;

int main(int argc, char **argv) {

    // RANDOM GENERATOR INITIALIZATION
    RandomGenerator     g;

    if (argc < 2) {
        cout << "\tERROR: input_sim_yaml not passed as command line argument" << endl;
        exit(1);
    }
    cout << "Input sim file: ";
    cout << argv[1] << "\n" << endl;

    string              input_sim_yaml = argv[1];
    ifstream            f_in(input_sim_yaml);
    YAML::Node          sim_par = YAML::Load(f_in);
    string              s, input_mode_config;
    bool                status;
    string              continue_s;

    cout << "Simulation is going to start with the following parameters:" << endl;
    cout << "t_end\t\t\t" << sim_par["t_end"].as<double>() << endl;
    cout << "dt\t\t\t" << sim_par["dt"].as<double>() << endl;
    cout << "input_mode\t\t" << sim_par["input_mode"].as<unsigned>() << endl;
    cout << "n_step\t\t\t" << sim_par["n_step"].as<unsigned>() << endl;
    cout << "out_dir\t\t\t" << sim_par["out_dir"].as<string>() << endl;
    cout << "subnets_config_yaml\t" << sim_par["subnets_config_yaml"].as<string>() << endl;
    cout << "weights_config_yaml\t" << sim_par["weights_config_yaml"].as<string>() << endl;
    cout << "connections_config_yaml\t" << sim_par["connections_config_yaml"].as<string>() << endl;
    cout << "to_save_config_yaml\t" << sim_par["to_save_config_yaml"].as<string>() << endl;
    if ( sim_par["input_mode"].as<unsigned>()==2 ) {
        cout << "input_mode_config\t" << sim_par["input_mode_config"].as<string>() << endl;
        cout << "pay attention if you parallelise: input_mode_config file is read with a bit of delay!" << endl;
        exit(1);
        input_mode_config = sim_par["input_mode_config"].as<string>();
    }
    else {
        input_mode_config = "";
    }

    s = sim_par["out_dir"].as<string>();
    status = mkdir(s.c_str(), 0755);
        // status 0: directory correctly created
        // status 1: directory already existing or path not ok
    if (status == 1) {
        cout << "\tNote: wrong or already existing out_dir [" << s << "]\nContinue? [y or n]: ";
        getline (cin, continue_s);
        if (continue_s=="n") {
            cout << "Terminating..." <<endl;
            printf ("Error creating dir: %s\n",strerror(errno));
            exit(1);
        }
        else if (continue_s=="y") {
            system( ("rm -r "+sim_par["out_dir"].as<string>()).c_str() );
            status = mkdir(s.c_str(), 0755);
            if (status == 1) {
                cout << "\tERROR: Something went wrong creating the output directory: " << sim_par["out_dir"].as<string>() <<endl;
                exit(1);
            }
        }
        else {
            cout << "\tERROR: non valid input: " << continue_s << endl;
            exit(1);
        }
    }

    cout << "\tricorda che così funziona solo se i config_files sono subito dentro una cartella" << endl;

    // copy config files in out_dir
    system(("cp ./" + sim_par["subnets_config_yaml"].as<string>() + " ./" + sim_par["out_dir"].as<string>() ).c_str());
    system(("cp ./" + sim_par["weights_config_yaml"].as<string>() + " ./" + sim_par["out_dir"].as<string>() ).c_str());
    system(("cp ./" + sim_par["connections_config_yaml"].as<string>() + " ./" + sim_par["out_dir"].as<string>() ).c_str());
    system(("cp ./" + sim_par["to_save_config_yaml"].as<string>() + " ./" + sim_par["out_dir"].as<string>() ).c_str());
    system(("cp " + input_sim_yaml + " ./" + sim_par["out_dir"].as<string>() ).c_str());

    cout << "\tconfig files letti e copiati" << endl;

    Network Net(sim_par["t_end"].as<double>(), sim_par["n_step"].as<unsigned>(),  \
                sim_par["dt"].as<double>(), sim_par["input_mode"].as<unsigned>(), \
                join_and_correct_config(sim_par["subnets_config_yaml"].as<string>(), sim_par["out_dir"].as<string>()),      \
                join_and_correct_config(sim_par["weights_config_yaml"].as<string>(), sim_par["out_dir"].as<string>()),      \
                join_and_correct_config(sim_par["connections_config_yaml"].as<string>(), sim_par["out_dir"].as<string>()),  \
                join_and_correct_config(sim_par["to_save_config_yaml"].as<string>(), sim_par["out_dir"].as<string>()),      \
                sim_par["out_dir"].as<string>(), g, input_mode_config );

}
