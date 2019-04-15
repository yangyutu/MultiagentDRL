


#include "GridWorld.h"


GridWorld::GridWorld(std::string configName0, int randomSeed0){
    
    std::ifstream ifile(configName0);
    ifile >> config;
    ifile.close();
    this->simulator = std::shared_ptr<ParticleSimulator>(new ParticleSimulator(configName0, randomSeed0));
    read_config();
    read_map();
    initialize();
    
}


void GridWorld::read_config(){
    numP = config["N"];
    n_channel = config["n_channel"];
    receptHalfWidth = config["receptHalfWidth"];
    radius = config["radius"];
    n_step = config["GridWorldNStep"];
    rewardShareThresh = config["rewardShareThresh"];
    rewardShareCoeff = config["rewardShareCoeff"];
    obstaclePadding = config["obstaclePadding"];
}

void GridWorld::step_cpp(std::vector<int>& actions) {
    simulator->run(n_step, actions);
}

void GridWorld::step(py::array_t<int>& actions){
    
    auto buf = actions.request();
    int *ptr = (int *)buf.ptr;
    int size = buf.size;
    std::vector<int> actions_cpp(ptr,  ptr + size);
    
    simulator->run(n_step, actions_cpp);
}


void GridWorld::reset(){
    simulator->createInitialState();
}

void GridWorld::initialize(){
    
    
    for (int i = -receptHalfWidth; i < receptHalfWidth+1; i++){
        for (int j = -receptHalfWidth; j < receptHalfWidth+1; j++){
            sensorXIdx.push_back(i);
            sensorYIdx.push_back(j);
        }
    }
    sensorArraySize = sensorXIdx.size();
    sensorWidth = 2*receptHalfWidth + 1;
    
    linearSensor_Previous.resize(numP*sensorArraySize);
    std::fill(linearSensor_Previous.begin(), linearSensor_Previous.end(), 0);
}

void GridWorld::read_map(){
    
    std::string line;
    std::vector<std::vector<int> > mapData;
    std::string mapName =  config["mapName"];
    std::ifstream mapfile ( mapName + ".txt");
    if( mapfile.is_open() ){ while( mapfile.good() ){
            while(!std::getline(mapfile, line, '\n').eof()) {

                    std::istringstream reader(line);
                    std::vector<int> lineData;
                    std::string::const_iterator i = line.begin();
                    while(!reader.eof()) {
                        double val;
                        reader >> val;
                        if(reader.fail()) break;
                        lineData.push_back(val);
            }
            mapData.push_back(lineData);
            }
    }}
    else{ std::cout<< "Unable to open file." << std::endl; }

    mapfile.close();
    mapRows = mapData.size();
    mapCols = mapData[0].size();
    
    std::vector<std::vector<int> > obsMap; 
    for (int i = 0; i < mapRows + 2*obstaclePadding; i++){
        std::vector<int> line(mapCols + 2*obstaclePadding, 3);
        if ((i >= obstaclePadding) && (i < mapRows + obstaclePadding) )
        {
            for (int j = obstaclePadding; j < mapCols + obstaclePadding; j++){
                line[j] = mapData[i - obstaclePadding][j - obstaclePadding];
            }
        }
        obsMap.push_back(line);
    }

    std::ofstream mapOut(mapName + "out.txt");
    
    obsMapRows = obsMap.size();
    obsMapCols = obsMap[0].size();
    
    for(int i = 0; i < obsMapRows; i++){
        for(int j = 0; j < obsMapCols; j++){
            mapOut << obsMap[i][j] <<" ";
            CoorPair cp(i,j);
            MapSlot slot(i,j);
            
            if(obsMap[i][j] == 1){
                slot.occupiedByFood = true;  
            }else if(obsMap[i][j] == 2){
                slot.occupiedByHazard = true;
            }else if(obsMap[i][j] == 3){
                slot.occupiedByObstacle= true;
            }
            mapInfo[cp] = slot;
        }
        mapOut << std::endl;
    }
    mapOut.close();
}

void GridWorld::fill_observation(const ParticleSimulator::partConfig& particles, std::vector<int>&  linearSensorAll){

    // fill map position with particle occupation
    for (int i = 0; i < numP; i++)
    {
        int x_int = (int)std::floor(particles[i]->r[0]/radius + 0.5) + obstaclePadding;
        int y_int = (int)std::floor(particles[i]->r[1]/radius + 0.5) + obstaclePadding;
        mapInfo[CoorPair(x_int, y_int)].occupiedByParticle = true;
    }
    int idx1, idx2, idx3;
    for (int i = 0; i < numP; i++)
    {
        double phi = particles[i]->phi;
        
        
        for(int j = 0; j < sensorArraySize; j++){
            // transform from local to global
            double x = sensorXIdx[j]*cos(phi) - sensorYIdx[j]*sin(phi) + particles[i]->r[0]/radius;
            double y = sensorXIdx[j]*sin(phi) + sensorYIdx[j]*cos(phi) + particles[i]->r[1]/radius;
            int x_int = (int)std::floor(x + 0.5) + obstaclePadding;
            int y_int = (int)std::floor(y + 0.5) + obstaclePadding; 
        
            MapSlot slot = mapInfo[CoorPair(x_int, y_int)];
            
                
            idx1 = i*n_channel*sensorArraySize + j;
            idx2 = i*n_channel*sensorArraySize + j + sensorArraySize;
            idx3 = i*n_channel*sensorArraySize + j + sensorArraySize*2;
            //if (slot.occupiedByFood || slot.occupiedByHazard){
            //    linearSensorAll[idx1] = 1;
            //}else if (slot.occupiedByParticle){
             //   linearSensorAll[idx2] = 1;
            //}
            if (slot.occupiedByParticle){
                linearSensorAll[idx2] = 1;
                    //exclude self
                if (sensorXIdx[j] == 0 && sensorYIdx[j] ==0){
                    linearSensorAll[idx2] = 0;
                }
            }
            
            linearSensorAll[idx3] = linearSensor_Previous[i*sensorArraySize + j];
            linearSensor_Previous[i*sensorArraySize + j] = linearSensorAll[idx2];
            
        }
    }

    // reset information
    for (int i = 0; i < numP; i++)
    {
        int x_int = (int)std::floor(particles[i]->r[0]/radius + 0.5) + obstaclePadding;
        int y_int = (int)std::floor(particles[i]->r[1]/radius + 0.5) + obstaclePadding;
        mapInfo[CoorPair(x_int, y_int)].occupiedByParticle = false;
    }
    
#ifdef DEBUG
    for (int i = 0; i < numP; i++)
    {
        std::cout << "particle:" + std::to_string(i) << std::endl;
        for(int n = 0; n < n_channel; n++){
            std::cout << "channel:" + std::to_string(n) << std::endl;
            for(int j = 0; j < sensorWidth; j++){
                for(int k = 0; k < sensorWidth; k++){
                    std::cout << linearSensorAll[i*n_channel*sensorArraySize+n*sensorArraySize + j*sensorWidth + k ] << " ";
                }
                std::cout << "\n";
            }
        }
        
    }
#endif
    

}





std::vector<int> GridWorld::get_observation_cpp(){

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(numP*n_channel*sensorArraySize, 0);

    fill_observation(particles, linearSensorAll);
    
    

    return linearSensorAll;
}

py::array_t<int> GridWorld::get_observation(){
    
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(numP*n_channel*sensorArraySize, 0);

    fill_observation(particles, linearSensorAll);
    
    py::array_t<int> result(numP*n_channel*sensorArraySize, linearSensorAll.data());

    return result;
}


std::vector<double> GridWorld::cal_reward_cpp(){
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    std::vector<double> rewards(numP, 0.0);
    for(int i = 0; i < numP; i++){
        int x_int = (int)std::floor(particles[i]->r[0]/radius + 0.5) + obstaclePadding;
        int y_int = (int)std::floor(particles[i]->r[1]/radius + 0.5) + obstaclePadding;
        MapSlot slot = mapInfo[CoorPair(x_int, y_int)];
        if (slot.occupiedByFood){
            rewards[i] = 1;
        }
        if (slot.occupiedByHazard){
            rewards[i] = -1;
        }
    }
    
    std::vector<double> rewardIndividual = rewards;
    
    double r[2];
    for(int i = 0; i < numP - 1; i++){
        for(int j = i + 1; j < numP; j++){
            r[0] = (particles[i]->r[0] - particles[j]->r[0])/radius;
            r[1] = (particles[i]->r[1] - particles[j]->r[1])/radius;
            double dist = sqrt(r[0]*r[0] + r[1]*r[1]);
            if (dist < rewardShareThresh){
                rewards[i] += rewardShareCoeff*rewards[j];
                rewards[j] += rewardShareCoeff*rewards[i];
              
            }
        }
    }
    
    
    return rewards;
}

py::array_t<double> GridWorld::cal_reward(){
    
    std::vector<double> rewards = cal_reward_cpp();
    py::array_t<double> result(numP, rewards.data());

    return result;
}

py::array_t<double> GridWorld::get_positions(){
    std::vector<double> positions(3*numP);
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    for (int i = 0; i < numP; i++) {
        positions[i*3] =  particles[i]->r[0]/radius;
        positions[i*3 + 1] =  particles[i]->r[1]/radius;
        positions[i*3 + 2] =  particles[i]->phi;
    }
    
    py::array_t<double> result(3*numP, positions.data());

    return result;
}

void GridWorld::set_iniConfigs(py::array_t<double> iniConfig){
    auto buf = iniConfig.request();
    double *ptr = (double *)buf.ptr;
    int size = buf.size;
    std::vector<double> iniConfig_cpp(ptr,  ptr + size);
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    for (int i = 0; i < numP; i++) {
        particles[i]->r[0] = iniConfig_cpp[i*3]*radius;
        particles[i]->r[1] = iniConfig_cpp[i*3 + 1]*radius;
        particles[i]->phi = iniConfig_cpp[i*3 + 2];
    }

}