function [Actor, Critic] = PPOUpdate(Actor, Critic, buffer)
% PPOUpdate
% Update Actor and Critic using PPO clipped objective
%
% Requirements:
%   - Actor.net      : dlnetwork
%   - Critic.net     : dlnetwork
%   - buffer contains:
%       states, actions, rewards, log_probs, values, dones
%
% Notes:
%   - This version uses full-batch PPO update
%   - First version uses simple returns/advantages, not GAE

    % ===== hyperparameters =====
    gamma       = 0.99;
    clipEps     = 0.2;
    entropyCoef = 0.01;
    valueCoef   = 0.5;
    ppoEpoch    = 3;
    maxGradNorm = 0.5;

    % ===== compute returns and advantages (GAE) =====
    lambda = 0.95;

    T = size(buffer.next_states,1);
    next_values = zeros(T,1);

    for t = 1:T
        next_values(t) = CriticForward(Critic, buffer.next_states(t,:));
    end

    [returns, advantages] = ComputeReturnsAndAdvantages( ...
        buffer.rewards, buffer.values, next_values, buffer.dones, gamma, lambda);

    % ===== prepare data =====
    states      = single(buffer.states);       % T x stateDim
    actions     = single(buffer.actions);      % T x actionDim
    oldLogProbs = single(buffer.log_probs);    % T x 1
    returns     = single(returns);             % T x 1
    advantages  = single(advantages);          % T x 1

    % ===== initialize optimizer states if needed =====
    if ~isfield(Actor, 'step') || isempty(Actor.step)
        Actor.step = 0;
    end
    if ~isfield(Critic, 'step') || isempty(Critic.step)
        Critic.step = 0;
    end
    if isempty(Actor.avgGrad),        Actor.avgGrad = [];        end
    if isempty(Actor.avgSqGrad),      Actor.avgSqGrad = [];      end
    if isempty(Critic.avgGrad),       Critic.avgGrad = [];       end
    if isempty(Critic.avgSqGrad),     Critic.avgSqGrad = [];     end

    % ===== PPO epochs =====
    for epoch = 1:ppoEpoch

        % ----- Actor update -----
        [actorLoss, actorGradients] = dlfeval( ...
            @ActorLoss, Actor.net, ...
            states, actions, oldLogProbs, advantages, clipEps, entropyCoef);

        actorGradients = ClipGradientTable(actorGradients, maxGradNorm);

        Actor.step = Actor.step + 1;

        [Actor.net, Actor.avgGrad, Actor.avgSqGrad] = adamupdate( ...
            Actor.net, actorGradients, ...
            Actor.avgGrad, Actor.avgSqGrad, ...
            Actor.step, Actor.learnRate);

        % ----- Critic update -----
        [criticLoss, criticGradients] = dlfeval( ...
            @CriticLoss, Critic.net, states, returns, valueCoef);

        criticGradients = ClipGradientTable(criticGradients, maxGradNorm);

        Critic.step = Critic.step + 1;

        [Critic.net, Critic.avgGrad, Critic.avgSqGrad] = adamupdate( ...
            Critic.net, criticGradients, ...
            Critic.avgGrad, Critic.avgSqGrad, ...
            Critic.step, Critic.learnRate);

        % 可选：打印 loss 观察训练
        % fprintf('PPO epoch %d | actor loss = %.6f | critic loss = %.6f\n', ...
        %     epoch, double(gather(extractdata(actorLoss))), double(gather(extractdata(criticLoss))));
    end
end


% =========================================================
% Actor loss
% =========================================================
function [loss, gradientsNet] = ActorLoss(net, ...
    states, actions, oldLogProbs, advantages, clipEps, entropyCoef)

    % states:      T x stateDim
    % actions:     T x actionDim
    % oldLogProbs: T x 1
    % advantages:  T x 1

    batchSize = size(states,1);
    actionDim = size(actions,2);

    % ---- convert to dlarray ----
    dlStates      = dlarray(states', "CB");          % stateDim x T
    dlActions     = dlarray(actions', "CB");         % actionDim x T
    dlOldLogProbs = dlarray(oldLogProbs', "CB");     % 1 x T
    dlAdvantages  = dlarray(advantages', "CB");      % 1 x T

    % ---- forward ----
    [dlActionMean, dlActionLogStd] = forward(net, dlStates, Outputs={'actionMean','actionLogStd'});
    dlActionLogStd = max(min(dlActionLogStd, -0.5), -4);
    dlStdMat = exp(dlActionLogStd);
    dlStdMat = max(dlStdMat, 1e-6);
    dlVarMat = dlStdMat .^ 2;

    % ---- new log prob ----
    dlLogProb = -0.5 * sum(((dlActions - dlActionMean).^2) ./ dlVarMat + log(2*pi*dlVarMat), 1);  % 1 x T

    % ---- entropy ----
    dlEntropy = 0.5 * sum(log(2*pi*exp(1)*dlVarMat), 1);   % 1 x T

    % ---- PPO ratio ----
    ratio = exp(dlLogProb - dlOldLogProbs);   % 1 x T

    % ---- clipped objective ----
    surr1 = ratio .* dlAdvantages;
    surr2 = min(max(ratio, 1 - clipEps), 1 + clipEps) .* dlAdvantages;

    % maximize min(surr1,surr2) + entropy bonus
    objective = min(surr1, surr2) + entropyCoef * dlEntropy;

    loss = -mean(objective);

    % ---- gradients ----
    gradientsNet = dlgradient(loss, net.Learnables);
end


% =========================================================
% Critic loss
% =========================================================
function [loss, gradients] = CriticLoss(net, states, returns, valueCoef)

    % states:  T x stateDim
    % returns: T x 1

    dlStates  = dlarray(states', "CB");          % stateDim x T
    dlReturns = dlarray(returns', "CB");         % 1 x T

    dlValues = forward(net, dlStates);           % 1 x T

    loss = valueCoef * mean((dlValues - dlReturns).^2);

    gradients = dlgradient(loss, net.Learnables);
end

function gradients = ClipGradientTable(gradients, maxGradNorm)

    totalNorm = sqrt(GradientTableNormSq(gradients));
    if totalNorm > maxGradNorm
        scale = maxGradNorm / (totalNorm + 1e-12);
        gradients = ScaleGradientTable(gradients, scale);
    end
end

function totalNormSq = GradientTableNormSq(gradients)

    totalNormSq = 0;
    for k = 1:size(gradients,1)
        g = gradients.Value{k};
        if ~isempty(g)
            totalNormSq = totalNormSq + sum(extractdata(g).^2, 'all');
        end
    end
end

function gradients = ScaleGradientTable(gradients, scale)

    for k = 1:size(gradients,1)
        if ~isempty(gradients.Value{k})
            gradients.Value{k} = gradients.Value{k} .* scale;
        end
    end
end
