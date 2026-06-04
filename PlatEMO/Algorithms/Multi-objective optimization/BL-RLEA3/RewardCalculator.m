function [reward,stats] = RewardCalculator( ...
    Problem,Offspring,Nsample,ppoSurvived,deltaPPO,perturbScale)
% Paired counterfactual reward for PPO perturbations.
%
% Offspring order must be [baseEA; perturbedEA].
% The reward compares baseEA(i) with perturbedEA(i), so ordinary SBX+PM
% variation does not receive credit that should belong only to PPO.

    if Nsample <= 0
        reward = 0;
        stats.betterRate = 0;
        stats.meanGain = 0;
        stats.bestGain = 0;
        stats.survivalRate = 0;
        stats.meanStep = 0;
        return;
    end

    BaseOffspring = Offspring(1:Nsample);
    PPOOffspring  = Offspring(Nsample+1:2*Nsample);

    baseFit = CalFitness(Problem.C,BaseOffspring);
    ppoFit  = CalFitness(Problem.C,PPOOffspring);

    baseFit = baseFit(:);
    ppoFit  = ppoFit(:);

    gain = baseFit - ppoFit;

    if nargin < 4 || isempty(ppoSurvived)
        ppoSurvived = false(Nsample,1);
    end
    ppoSurvived = logical(ppoSurvived(:));

    if nargin < 5 || isempty(deltaPPO)
        deltaPPO = zeros(Nsample,Problem.DU);
    end
    if nargin < 6 || isempty(perturbScale)
        perturbScale = ones(1,Problem.DU);
    end

    perturbScale = perturbScale(:)';
    scaleMat = repmat(perturbScale,Nsample,1);
    stepSize = mean(abs(deltaPPO ./ (scaleMat + 1e-12)),2);

    betterRate = mean(gain > 0);
    meanGain = mean(gain);
    medianGain = median(gain);
    bestGain = min(baseFit) - min(ppoFit);

    pairScale = abs(baseFit) + 1e-8;
    meanScale = mean(abs(baseFit)) + 1e-8;

    rBetterRate = 2 * betterRate - 1;
    rPairGain   = mean(tanh(gain ./ pairScale));
    rMeanGain   = tanh(meanGain ./ meanScale);

    % Keep the reward focused on paired counterfactual gains.  Survival is
    % tracked in stats, but not used as a learning signal here.
    reward = 0.50 * rBetterRate + ...
             0.30 * rPairGain + ...
             0.20 * rMeanGain;

    % Discourage very large perturbations when they do not buy paired gains.
    if meanGain <= 0
        reward = reward - 0.03 * mean(stepSize);
    end

    reward = max(min(reward,1),-1);

    stats.betterRate = betterRate;
    stats.meanGain = meanGain;
    stats.medianGain = medianGain;
    stats.bestGain = bestGain;
    stats.survivalRate = mean(ppoSurvived);
    stats.meanReward = reward;
    stats.meanStep = mean(stepSize);
    stats.baseBest = min(baseFit);
    stats.ppoBest = min(ppoFit);
    stats.baseMean = mean(baseFit);
    stats.ppoMean = mean(ppoFit);
end
