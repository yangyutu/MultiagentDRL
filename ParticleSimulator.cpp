#include "ParticleSimulator.h"


double const ParticleSimulator::T = 293.0;
double const ParticleSimulator::kb = 1.38e-23;
double const ParticleSimulator::vis = 1e-3;



ParticleSimulator::ParticleSimulator(std::string configName0, int randomSeed0){
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);
    randomSeed = randomSeed0;
    configName = configName0;
    std::ifstream ifile(this->configName);
    ifile >> config;
    ifile.close();

    readConfigFile();
    
    
    // cellListFlag = false;
    // if (parameter.particleCellListFlag){
    //     this->cellList = std::make_shared<CellList>(parameter.cellListCutoff*radius,
    //             parameter.cellListDim,parameter.cellListMaxCount,
    //             parameter.cellListBox_x*radius, parameter.cellListBox_y*radius,
    //             parameter.cellListBox_z*radius);
    //     cellListFlag = true;
    // }
    // if (parameter.obstacleCellListFlag){
    //     this->obsCellList = std::make_shared<CellList>(parameter.cellListCutoff*radius,
    //             parameter.cellListDim,parameter.cellListMaxCount,
    //             parameter.cellListBox_x*radius, parameter.cellListBox_y*radius,
    //             parameter.cellListBox_z*radius);
    // }
    
    if (obstacleFlag) {
         this->readObstacle();
        // this->obsCellList = std::make_shared<CellList>(parameter.cellListCutoff*radius,
        //         parameter.cellListDim,parameter.cellListMaxCount,
        //         parameter.cellListBox_x*radius, parameter.cellListBox_y*radius,
        //         parameter.cellListBox_z*radius);
        // int builtCount = this->obsCellList->buildList(obstacles);
            
        //     if (builtCount!=numObstacles){
        //         std::cout << "build imcomplete" << std::endl;
        //     }

    }
    
    if (wallFlag){
        this->getWallInfo();
    }
    
    
    
    for(int i = 0; i < numP; i++){
        particles.push_back(particle_ptr(new ParticleSimulator::particle));
    }
}



void ParticleSimulator::readConfigFile(){

    randomMoveFlag = config["randomMoveFlag"];
    filetag = config["filetag"];
    iniFile = config["iniConfig"];
    maxSpeed = config["maxSpeed"];
    
    numP = config["N"];
    radius = config["radius"];
    dt_ = config["dt"];
    maxTurnSpeed = 15.0/180.0*3.1415926; // 1 s turn 15 degree
    diffusivity_r = 0.161; // characteristic time scale is about 6s
    diffusivity_t = 2.145e-13;// this corresponds the diffusivity of 1um particle
    //diffusivity_r = parameter.diffu_r; // this correponds to rotation diffusity of 1um particle
    Bpp = config["Bpp"];
    Bpp = Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
    Kappa = config["kappa"]; // here is kappa*radius
    //Os_pressure = config["Os_pressure"];
    //Os_pressure = Os_pressure * kb * T * 1e9;
    //L_dep = parameter.L_dep; // 0.2 of radius size, i.e. 200 nm
    radius_nm = radius*1e9;
    //combinedSize = (1+L_dep)*radius_nm;
    mobility = diffusivity_t/kb/T;
    trajOutputInterval = 1.0/dt_;
    fileCounter = 0;
    cutoff = config["cutoff"];
    //numControl = this->velocity.size();
    trajOutputFlag = config["trajOutputFlag"];
    obstacleFlag = config["obstacleFlag"];
    wallFlag = config["wallFlag"];
    constantPropelFlag = false;
    if(config.contains("constantPropel"))
        constantPropelFlag = config["constantPropel"];
    
    this->rand_generator.seed(randomSeed);




}

void ParticleSimulator::runHelper() {

    if (((this->timeCounter ) == 0) && trajOutputFlag) {
        this->outputTrajectory(this->trajOs);
    }

    calForces();

    
    for (int i = 0; i < numP; i++) {

        double randomX, randomY, randomPhi;
        randomX = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);
        randomY = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);
        randomPhi = sqrt(2.0 * diffusivity_r * dt_) * (*rand_normal)(rand_generator);
        
        particles[i]->r[0] += mobility * particles[i]->F[0] * dt_ +
                    particles[i]->u * cos(particles[i]->phi) * dt_;
        particles[i]->r[1] += mobility * particles[i]->F[1] * dt_ +
                    particles[i]->u * sin(particles[i]->phi) * dt_;
        particles[i]->phi += particles[i]->w * dt_;
        
        if (particles[i]->phi < 0){
            particles[i]->phi += 2*M_PI;
        }else if(particles[i]->phi > 2*M_PI){
            particles[i]->phi -= 2*M_PI;
        }
        
        if(randomMoveFlag){
            particles[i]->r[0] +=randomX;
            particles[i]->r[1] +=randomY;
            particles[i]->phi +=randomPhi;
        
        }
        
            
    }
        
    this->timeCounter++;
    if (((this->timeCounter ) % trajOutputInterval == 0) && trajOutputFlag) {
        this->outputTrajectory(this->trajOs);
    }
}

void ParticleSimulator::run(int steps, const std::vector<int>& actions){
    
    if (constantPropelFlag) {
        for (int i =0; i < numP; i++){
            particles[i]->action = actions[i];
            if(actions[i] == 1)
            {
                particles[i]->u = maxSpeed;
                particles[i]->w = maxTurnSpeed;
            }
            else if(actions[i] == 2)
            {
                particles[i]->u = maxSpeed;
                particles[i]->w = -maxTurnSpeed;
            }
            else
            {
                particles[i]->u = maxSpeed;
                particles[i]->w = 0;
            }
        }    
    
    } else {
     
        for (int i =0; i < numP; i++){
            particles[i]->action = actions[i];
            if(actions[i] == 1)
            {
                particles[i]->u = maxSpeed;
                particles[i]->w = 0;
            }
            else if(actions[i] == 2)
            {
                particles[i]->u = 0;
                particles[i]->w = maxTurnSpeed;
            }
            else if(actions[i] == 3)
            {
                particles[i]->u = 0;
                particles[i]->w = -maxTurnSpeed;
            }
            else
            {
                particles[i]->u = 0;
                particles[i]->w = 0;
            }
        }
    }
    
    for (int i = 0; i < steps; i++){
	    runHelper();
    }
}


// this force calculation includes double layer repulsion and depletion attraction 
void ParticleSimulator::calForcesHelper_DLAO(double ri[3], double rj[3], double F[3],int i,int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t"<< this->timeCounter << "dist: " << dist <<std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
        double Fpp = -4.0/3.0*
        Os_pressure*M_PI*(-3.0/4.0*pow(combinedSize,2.0)+3.0*dist*dist/16.0*radius_nm*radius_nm);
        Fpp += -Bpp * Kappa * exp(-Kappa*(dist-2.0));
        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;

        }
    }
}

// this force calculation only includes double layer repulsion 
void ParticleSimulator::calForcesHelper_DL(double ri[3], double rj[3], double F[3],int i, int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t with " << j << "\t"<< this->timeCounter << "dist: " << dist <<std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
        double Fpp = -Bpp * Kappa * exp(-Kappa*(dist-2.0));
        
        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;
        }
    }
}

void ParticleSimulator::calForces() {
    double r[dimP], dist, F[3];
    for (int i = 0; i < numP; i++) {
        for (int k = 0; k < dimP; k++) {
            particles[i]->F[k] = 0.0;
        }
    }


    for (int i = 0; i < numP - 1; i++) {
        for (int j = i + 1; j < numP; j++) {
            calForcesHelper_DL(particles[i]->r, particles[j]->r, F,i, j);
            for (int k = 0; k < dimP; k++) {
                particles[i]->F[k] += F[k];
                particles[j]->F[k] += -F[k];
            }
        }
    }
        
    if (wallFlag){
        for (int i = 0; i < numP; i++) {
            double dist_x = (particles[i]->r[0]/radius - wallX[0]);
            particles[i]->F[0] += 2.0*Bpp * Kappa * exp(-Kappa*(dist_x-1.0));
            dist_x = (wallX[1] - particles[i]->r[0]/radius);
            particles[i]->F[0] += -2.0*Bpp * Kappa * exp(-Kappa*(dist_x-1.0));
            
            double dist_y = (particles[i]->r[1]/radius - wallY[0]);
            particles[i]->F[1] += 2.0*Bpp * Kappa * exp(-Kappa*(dist_y-1.0));
            dist_y = (wallY[1] - particles[i]->r[1]/radius);
            particles[i]->F[1] += -2.0*Bpp * Kappa * exp(-Kappa*(dist_y-1.0));
            
        }
    }
        

       
    
    // if(config["obstacleFlag"]) {
    //     for (int i = 0; i < numP; i++) {
    //         std::vector<int> nblist =
    //                 obsCellList->getNeighbors(particles[i]->r[0], particles[i]->r[1], particles[i]->r[2]);
    //         int nblistSize = nblist.size();
    //         for (int j = 0; j < nblistSize; j++) {
    //             calForcesHelper_DL(particles[i]->r, obstacles[nblist[j]]->r, F, i, nblist[j]);
    //             for (int k = 0; k < dimP; k++) {
    //                 particles[i]->F[k] += F[k];
    //             }
    //         }

    //     }
   
    // }
    
}
    


void ParticleSimulator::createInitialState(){

    this->readxyz(iniFile);
    
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open() && trajOutputFlag ) trajOs.close();
    if (trajOutputFlag)
        this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
    this->timeCounter = 0;
    
}

void ParticleSimulator::close(){
    if (trajOs.is_open()) trajOs.close();
    

}

void ParticleSimulator::getWallInfo(){
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
    int mapRows = mapData.size();
    int mapCols = mapData[0].size();

    wallX[0] = 0;
    wallX[1] = mapRows;
    wallY[0] = 0;
    wallY[1] = mapCols;
}

void ParticleSimulator::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < numP; i++) {
        os << i << "\t";
        for (int j = 0; j < dimP; j++){
            os << particles[i]->r[j]/radius << "\t";
        }
        
        os << particles[i]->phi<< "\t";
        os << particles[i]->u / radius<< "\t";
        os << particles[i]->w<<"\t";
        os << particles[i]->action << "\t";
        os << std::endl;
    }
    
}



void ParticleSimulator::readxyz(const std::string filename) {
    std::ifstream is;
    is.open(filename.c_str());
    std::string line;
    double dum;
    for (int i = 0; i < numP; i++) {
        getline(is, line);
        std::stringstream linestream(line);
        linestream >> dum;
        linestream >> particles[i]->r[0];
        linestream >> particles[i]->r[1];
        linestream >> particles[i]->phi;
        particles[i]->u = 0.0;
        particles[i]->w = 0.0;
        
        if ((particles[i]->r[0] < wallX[0])||(particles[i]->r[0] > wallX[1])||(particles[i]->r[1] < wallY[0])||(particles[i]->r[1] > wallY[1])){
            std::cout << "particle outside wall!" << std::endl;
            exit(1);
        }
        
    }
    
    
    
    
    for (int i = 0; i < numP; i++) {
        particles[i]->r[0] *=radius;
        particles[i]->r[1] *=radius;
    }
    
    is.close();
}

void ParticleSimulator::readObstacle(){
    std::ifstream is;
    is.open(config["obstacleFilename"]);

    std::string line;
    double dum;
    double r[3];
    while (getline(is, line)){
        std::stringstream linestream(line);
        
        linestream >> dum;
        linestream >> r[0];
        linestream >> r[1];
        linestream >> r[2];
        obstacles.push_back(std::make_shared<pos>(r[0]*radius,r[1]*radius,r[2]*radius));
    } 
    numObstacles = obstacles.size();
    is.close();
}
