% function [returns, advantages] = ComputeReturnsAndAdvantages(rewards, dones, values, gamma)
% % ComputeReturnsAndAdvantages
% % Compute discounted returns and advantages for PPO
% %
% % Input:
% %   rewards    : T x 1 vector of rewards
% %   dones      : T x 1 vector, 1 means trajectory ends at this step
% %   values     : T x 1 vector of critic state values
% %   gamma      : discount factor, e.g. 0.99
% %
% % Output:
% %   returns    : T x 1 discounted returns
% %   advantages : T x 1 advantages = returns - values
% 
%     T = length(rewards);
% 
%     returns = zeros(T,1);
%     advantages = zeros(T,1);
% 
%     % ===== 1. compute returns =====
%     G = 0;
%     for t = T:-1:1
%         if dones(t) == 1
%             G = 0;
%         end
%         G = rewards(t) + gamma * G;
%         returns(t) = G;
%     end
% 
%     % ===== 2. compute advantages =====
%     advantages = returns - values;
% 
%     % ===== 3. normalize advantages =====
%     if std(advantages) > 1e-8
%         advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
%     else
%         advantages = advantages - mean(advantages);
%     end
% end
function [returns, advantages] = ComputeReturnsAndAdvantages( ...
    rewards, values, next_values, dones, gamma, lambda)
% ComputeReturnsAndAdvantages
% GAE version for PPO
%
% Input:
%   rewards     : T x 1
%   values      : T x 1
%   next_values : T x 1
%   dones       : T x 1
%   gamma       : discount factor
%   lambda      : GAE factor
%
% Output:
%   returns     : T x 1
%   advantages  : T x 1

    T = length(rewards);

    advantages = zeros(T,1);
    returns = zeros(T,1);

    gae = 0;

    for t = T:-1:1
        if dones(t) == 1
            delta = rewards(t) - values(t);
            gae = delta;
        else
            delta = rewards(t) + gamma * next_values(t) - values(t);
            gae = delta + gamma * lambda * gae;
        end

        advantages(t) = gae;
    end

    returns = advantages + values;

    % 标准化 advantage
    if std(advantages) > 1e-8
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
    else
        advantages = advantages - mean(advantages);
    end
end