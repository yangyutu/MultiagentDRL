

from Agents.MADQN.MADQN import MADQNAgent
from GridWorldEnv import GridWorldEnv
from utils.netInit import xavier_init

import json
from torch import optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import os

torch.manual_seed(1)
import torch.nn.functional as F
torch.set_num_threads(1)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
    
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(ConvNet, self).__init__()
        
        self.inputShape = (inputWidth,inputWidth)
        # 3 channel input
        self.layer1 = nn.Sequential( # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(2,             # input channel
                      32,            # output channel
                      kernel_size=2, # filter size
                      stride=1,
                      padding=1),   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2
        # add a fully connected layer
        #width = int(inputWidth / 4) + 1
        self.featureSize = self.featureSize()
        self.fc1 = nn.Linear(self.featureSize, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_action)
        self.apply(xavier_init)

    def forward(self, state):

        xout = self.layer1(state)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)

        out = F.relu(self.fc1(xout))
        out = self.fc2(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 2, *self.inputShape))).view(1, -1).size(1)



env = GridWorldEnv('config.json', 1)

N_S = env.stateDim[0]
N_A = env.nbActions


policyNet = ConvNet(N_S, 128, N_A)
targetNet = deepcopy(policyNet)
optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = MADQNAgent(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A)



trainFlag = True
testFlag = True

if trainFlag:

    if config['loadExistingModel']:
        checkpoint = torch.load(config['saveModelFile'])
        agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    plotPolicyFlag = False
    if plotPolicyFlag:

         for phiIdx in range(8):
             phi = phiIdx * np.pi/4.0
             policy = deepcopy(env.mapInfo).astype(np.long)
             value = deepcopy(env.mapInfo)
             for i in range(policy.shape[0]):
                   for j in range(policy.shape[1]):
                       #sensorInfo = env.agent.getSensorInfoFromPos(np.array([i, j, phi]))
                       distance = np.array(config['targetState']) - np.array([i, j])
                       dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
                       dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
                       dx = agent.env.agent.targetClipMap(dx) if dx > 0 else -agent.env.agent.targetClipMap(-dx)
                       dy = agent.env.agent.targetClipMap(dy) if dy > 0 else -agent.env.agent.targetClipMap(-dy)
                       #state = {'sensor': sensorInfo, 'target': np.array([dx, dy])}
                       #policy[i, j] = agent.getPolicy(state)
                       #Qvalue = agent.policyNet(stateProcessor([state], config['device'])[0])
                       #value[i, j] = Qvalue[0, policy[i,j]].cpu().item()
             np.savetxt('DynamicMazePolicyBeforeTrain' + config['mapName'] +'phiIdx'+ str(phiIdx) + '.txt', policy, fmt='%d', delimiter='\t')
             np.savetxt('DynamicMazeValueBeforeTrain' + config['mapName'] + 'phiIdx' + str(phiIdx) + '.txt', value, fmt='%.3f',delimiter='\t')
    # # plotPolicy(policy, N_A)



    agent.train()

    np.savetxt('posXSet.txt', agent.env.posXSet)
    np.savetxt('posYSet.txt', agent.env.posYSet)

    #    if plotPolicyFlag:
        # for phiIdx in range(8):
        #     phi = phiIdx * np.pi / 4.0
        #     policy = deepcopy(env.mapMat).astype(np.long)
        #     value = deepcopy(env.mapMat)
        #     for i in range(policy.shape[0]):
        #         for j in range(policy.shape[1]):
        #             if env.mapMat[i, j] == 1:
        #                 policy[i, j] = -1
        #                 value[i, j] = -1
        #             else:
        #                 sensorInfo = env.agent.getSensorInfoFromPos(np.array([i, j, phi]))
        #                 distance = np.array(config['targetState']) - np.array([i, j])
        #                 dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        #                 dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
        #                 dx = agent.env.agent.targetClipMap(dx) if dx > 0 else -agent.env.agent.targetClipMap(-dx)
        #                 dy = agent.env.agent.targetClipMap(dy) if dy > 0 else -agent.env.agent.targetClipMap(-dy)
        #                 state = {'sensor': sensorInfo, 'target': np.array([dx, dy])}
        #                 policy[i, j] = agent.getPolicy(state)
        #                 Qvalue = agent.policyNet(stateProcessor([state], config['device'])[0])
        #                 value[i, j] = Qvalue[0, policy[i,j]].cpu().item()
        #     np.savetxt('DynamicMazePolicyAfterTrain' + config['mapName'] + 'phiIdx' + str(phiIdx) + '.txt', policy, fmt='%d',
        #                delimiter='\t')
        #     np.savetxt('DynamicMazeValueAfterTrain' + config['mapName'] + 'phiIdx' + str(phiIdx) + '.txt', value,
        #                fmt='%.3f', delimiter='\t')

    torch.save({
                'model_state_dict': agent.policyNet.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                }, config['saveModelFile'])

testFlag = True
if testFlag:
    config['loadExistingModel'] = True

    if config['loadExistingModel']:
        checkpoint = torch.load(config['saveModelFile'])
        agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    config['trajOutputFlag'] = True
    config['filetag'] = 'Traj/Traj_test'
    config['dist_start_thresh'] = 0.7
    config['dist_end_thresh'] = 0.7
    with open('config_test.json', 'w') as fp:
        json.dump(config, fp)

    env = GridWorldEnv('config_test.json', 1)
    testStep = 100
    epIdx = 0
    print("start test!##################")
    runningAvgEpisodeReward = 0
    for _ in range(testStep):

        print("episode test index:" + str(epIdx))
        states = env.reset()
        rewardSum = 0


        for stepCount in range(agent.episodeLength):


            actions = agent.select_action(agent.policyNet, states, -0.1)

            nextStates, rewards, done, infos = env.step(actions)

            if stepCount == 0:
                print("at step 0:")
                print(infos)

            states = nextStates

            rewardSum += np.sum(rewards * math.pow(agent.gamma, stepCount))

            if done:
                break

        epIdx += 1
        runningAvgEpisodeReward = (runningAvgEpisodeReward * epIdx + rewardSum) / (epIdx + 1)
        print('episode test', epIdx)
        print("done in step count:")
        print(stepCount)
        print('rewards')
        print(rewards)
        print(infos)
        # print('reward sum: ', rewardSum)
        print("running average episode reward sum: {}".format(runningAvgEpisodeReward))

    np.savetxt('posXSet_test.txt', env.posXSet)
    np.savetxt('posYSet_test.txt', env.posYSet)

#    recorder = TrajRecorder()
#    agent.env.agent.config['stochMoveFlag'] = True
#    agent.testPolicyNet(100, recorder)
#    recorder.write_to_file(config['mapName'] + 'TestTraj.txt')

#plotPolicy(policy, N_A)
