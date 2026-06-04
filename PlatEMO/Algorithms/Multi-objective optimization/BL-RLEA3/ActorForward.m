% function [action, log_prob, action_mean, action_std, cache] = GetActionFromPPO(state, Actor)
% 
%     % ===== forward =====
%     z1 = state * Actor.W1 + Actor.b1;
%     h1 = tanh(z1);
% 
%     action_mean = h1 * Actor.W2 + Actor.b2;   % 1 x action_dim
%     action_std  = exp(Actor.log_std);         % 1 x action_dim
% 
%     % ===== sample from Gaussian =====
%     eps = randn(size(action_mean));
%     action = action_mean + action_std .* eps;
% 
%     % ===== log probability =====
%     var = action_std.^2;
%     log_prob = -0.5 * sum(((action - action_mean).^2) ./ var + log(2*pi*var));
% 
%     % ===== cache =====
%     cache.state = state;
%     cache.z1 = z1;
%     cache.h1 = h1;
%     cache.action_mean = action_mean;
%     cache.action_std = action_std;
% end
function [action, logProb, actionMean, actionStd] = ActorForward(Actor, state)
% ActorForward
% Forward pass of Gaussian actor network and sample action
%
% Input:
%   Actor      : actor struct from InitializeActorNetwork
%   state      : 1 x stateDim numeric vector
%
% Output:
%   action     : 1 x actionDim sampled action
%   logProb    : scalar log probability of sampled action
%   actionMean : 1 x actionDim mean of Gaussian policy
%   actionStd  : 1 x actionDim std of Gaussian policy

    % ===== 1. state -> dlarray =====
    % dlnetwork expects feature-by-batch for featureInputLayer
    dlState = dlarray(single(state(:)), "CB");

    % ===== 2. forward actor network to get action mean and state-dependent std =====
    [dlActionMean, dlActionLogStd] = forward(Actor.net, dlState, Outputs={'actionMean','actionLogStd'});
    dlActionLogStd = max(min(dlActionLogStd, -0.5), -4);

    actionMean = extractdata(dlActionMean)';         % 1 x actionDim
    actionStd = exp(extractdata(dlActionLogStd))';   % 1 x actionDim

    % avoid numerical issues
    actionStd = max(actionStd, 1e-6);

    % ===== 4. sample action from Gaussian =====
    eps = randn(1, Actor.actionDim);
    action = actionMean + actionStd .* eps;

    % ===== 5. compute log probability =====
    % For diagonal Gaussian:
    % log pi(a|s) = -0.5 * sum( ((a-mu)^2 / sigma^2) + log(2*pi*sigma^2) )
    var = actionStd .^ 2;
    logProb = -0.5 * sum(((action - actionMean).^2) ./ var + log(2*pi*var));
end
