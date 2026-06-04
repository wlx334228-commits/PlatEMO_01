% function Actor = InitializeActor(state_dim, action_dim, hidden_dim, lr)
% 
%     Actor.W1 = 0.1 * randn(state_dim, hidden_dim);
%     Actor.b1 = zeros(1, hidden_dim);
% 
%     Actor.W2 = 0.1 * randn(hidden_dim, action_dim);
%     Actor.b2 = zeros(1, action_dim);
% 
%     % 连续动作的对数标准差，先设成可学习参数
%     Actor.log_std = zeros(1, action_dim);
% 
%     Actor.lr = lr;
% end
function Actor = InitializeActorNetwork(stateDim, actionDim, hiddenDim, learnRate)
% InitializeActorNetwork
% Initialize Gaussian policy Actor network for PPO
%
% Input:
%   stateDim   : dimension of state
%   actionDim  : dimension of action
%   hiddenDim  : hidden layer width
%   learnRate  : learning rate
%
% Output:
%   Actor      : struct containing actor network and parameters

    % ===== Shared trunk + two heads: mean and state-dependent logStd =====
    trunk = [
        featureInputLayer(stateDim, Normalization="none", Name="state")
        fullyConnectedLayer(hiddenDim, Name="fc1")
        tanhLayer(Name="tanh1")
        fullyConnectedLayer(hiddenDim, Name="fc2")
        tanhLayer(Name="tanh2")
    ];

    meanHead = fullyConnectedLayer(actionDim, Name="actionMean");
    logStdHead = fullyConnectedLayer(actionDim, Name="actionLogStd");

    lgraph = layerGraph(trunk);
    lgraph = addLayers(lgraph, meanHead);
    lgraph = addLayers(lgraph, logStdHead);
    lgraph = connectLayers(lgraph, "tanh2", "actionMean");
    lgraph = connectLayers(lgraph, "tanh2", "actionLogStd");
    net = dlnetwork(lgraph);

    % ===== Store Actor =====
    Actor.net = net;

    % dimensions
    Actor.stateDim = stateDim;
    Actor.actionDim = actionDim;
    Actor.hiddenDim = hiddenDim;

    % optimizer related
    Actor.learnRate = learnRate;

    % Adam states
    Actor.avgGrad = [];
    Actor.avgSqGrad = [];
end
