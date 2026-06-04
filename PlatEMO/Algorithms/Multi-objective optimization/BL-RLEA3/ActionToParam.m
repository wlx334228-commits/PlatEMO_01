function Theta = ActionToParam(action,Problem,Population,sigma0)
% Convert the actor action into an additive perturbation distribution.
%
% action(1:DU)      controls the mean of the perturbation.
% action(DU+1:2DU) controls the standard deviation of the perturbation.

    PopDec = Population.decs;
    ulPop  = PopDec(:,1:Problem.DU);

    [~,DU] = size(ulPop);

    if length(action) ~= 2*DU
        error('ActionToParam: action dimension mismatch. Expected %d, got %d.',2*DU,length(action));
    end

    popStd = std(ulPop,0,1);
    popStd(popStd < 1e-6) = 1e-6;

    minSigma = AdaptiveMinSigma(Problem,Population,DU);
    popStd = max(popStd,minSigma);

    rawDeltaMu    = action(1:DU);
    rawDeltaSigma = action(DU+1:2*DU);

    perturbMu = 0.25 * tanh(rawDeltaMu) .* popStd;

    % This is an additive perturbation, so keep its initial scale smaller
    % than the base elite-sampling distribution.
    perturbLogSigma = -1.2 + 0.8 * tanh(rawDeltaSigma);
    perturbSigma = sigma0 .* exp(perturbLogSigma) .* popStd;

    Theta.mu = perturbMu;
    Theta.sigma = max(perturbSigma,1e-6);
end

function minSigma = AdaptiveMinSigma(Problem,Population,DU)

    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    fitness = CalFitness(Problem.C,Population);
    bestAbs = abs(min(fitness));

    if bestAbs > 1e-2
        scale = 0.02;
    elseif bestAbs > 1e-4
        scale = 0.005;
    else
        scale = 1e-5;
    end

    minSigma = max(scale .* range,1e-6);
end
