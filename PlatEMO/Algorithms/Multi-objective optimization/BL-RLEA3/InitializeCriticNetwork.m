% function Critic = InitializeCritic(state_dim, hidden_dim, lr)
% 
%     Critic.W1 = 0.1 * randn(state_dim, hidden_dim);
%     Critic.b1 = zeros(1, hidden_dim);
% 
%     Critic.W2 = 0.1 * randn(hidden_dim, 1);
%     Critic.b2 = 0;
% 
%     Critic.lr = lr;
% end
function Critic = InitializeCriticNetwork(stateDim, hiddenDim, learnRate)
% InitializeCriticNetwork
% Initialize Critic network for PPO
%
% Input:
%   stateDim   : dimension of state
%   hiddenDim  : hidden layer width
%   learnRate  : learning rate
%
% Output:
%   Critic     : struct containing critic network and parameters

    % ===== Critic network: state -> value =====
    layers = [
        featureInputLayer(stateDim, Normalization="none", Name="state")
        fullyConnectedLayer(hiddenDim, Name="fc1")
        tanhLayer(Name="tanh1")
        fullyConnectedLayer(hiddenDim, Name="fc2")
        tanhLayer(Name="tanh2")
        fullyConnectedLayer(1, Name="value")
    ];

    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);

    % ===== Store Critic =====
    Critic.net = net;

    % dimensions
    Critic.stateDim = stateDim;
    Critic.hiddenDim = hiddenDim;

    % optimizer related
    Critic.learnRate = learnRate;

    % Adam states (reserved for later PPO update)
    Critic.avgGrad = [];
    Critic.avgSqGrad = [];
end