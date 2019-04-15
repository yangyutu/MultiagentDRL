#include "ParticleSimulator.h"
#include "GridWorld.h"

void testSim(){
    ParticleSimulator simulator("config.json", 1);


    int step = 1000;
    simulator.createInitialState();
    

    for(auto i = 0; i < step; ++i){
        if(i%2 == 0){
            std::vector<int> actions(simulator.numP, 1);
            simulator.run(100, actions);
        }else{
            std::vector<int> actions(simulator.numP, 2);
            simulator.run(100, actions);
        }
    }
    simulator.close();

}
void testGridWorld(){
    
    int step = 100;
    GridWorld gw("config.json", 1);
    gw.simulator->config["filetag"] = "Traj/GridWorld";
    gw.simulator->readConfigFile();
    
    gw.reset();
    gw.get_observation_cpp();
    for(auto i = 0; i < step; ++i){
        std::cout << "step: " << i << std::endl;
        if(i%2 == 0){
            std::vector<int> actions(gw.numP, 1);
            gw.step_cpp(actions);
        }else{
            std::vector<int> actions(gw.numP, 2);
            gw.step_cpp(actions);
        }
        std::cout << "particle_phi: " << gw.simulator->getCurrState()[0]->phi << std::endl;
        gw.get_observation_cpp();
        std::vector<double> rewards = gw.cal_reward_cpp();
        for(auto j = 0; j < rewards.size(); j++)
            std::cout << rewards[j] << std::endl;
    }

    std::cout << "done print" << std::endl;
}


int main(){

    testGridWorld();
    
    



}