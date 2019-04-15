
/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11']
cfg['linker_args'] = ['-L/opt/OpenBLAS/lib  -llapack -lblas  -pthread -no-pie']
cfg['sources'] = ['GridWorld.cpp', 'ParticleSimulator.cpp']
%>
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "GridWorld.h"
namespace py = pybind11;


PYBIND11_MODULE(GridWorldPython, m) {    
    py::class_<GridWorld>(m, "GridWorldPython")
        .def(py::init<std::string, int>())
        .def("reset", &GridWorld::reset)
        .def("step", &GridWorld::step)
    	.def("getObservation", &GridWorld::get_observation)
        .def("calRewards", &GridWorld::cal_reward)
        .def("getPositions", &GridWorld::get_positions)
        .def("setIniConfig", &GridWorld::set_iniConfigs);
}