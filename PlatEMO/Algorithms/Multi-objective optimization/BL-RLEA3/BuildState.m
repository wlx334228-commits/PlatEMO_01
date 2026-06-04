function state = BuildState(Problem,Population,oldBest,noImproveGen, ...
    prevPPOBetterRate,prevMeanSigma,prevMeanGain,prevMeanStep)

    PopDec = Population.decs;
    PopCon = Population.cons;

    ulPop = PopDec(:,1:Problem.DU);
    ulFit = CalFitness(Problem.C,Population);
    N = size(ulPop,1);

    if isempty(PopCon) || Problem.C <= 0
        ulCon = zeros(N,1);
    else
        ulCon = PopCon(:,1:Problem.C);
    end

    bestFit = min(ulFit);

    popMean = mean(ulPop,1);

    lowerU = Problem.lower(1:Problem.DU);
    rangeU = Problem.upper(1:Problem.DU) - Problem.lower(1:Problem.DU);
    rangeU(rangeU < 1e-12) = 1;

    if size(ulPop,1) == 1
        popStd = zeros(1,Problem.DU);
    else
        popStd = std(ulPop,0,1);
    end

    feasULRate = mean(all(ulCon <= 0,2));

    g1 = signedLog(bestFit);
    g2 = tanh((oldBest - bestFit) / (abs(oldBest) + 1e-8));
    g3 = tanh(noImproveGen / 5);
    g4 = 2 * feasULRate - 1;
    g5 = 2 * prevPPOBetterRate - 1;
    g6 = signedLog(prevMeanSigma / (mean(rangeU) + 1e-12));
    g7 = tanh(prevMeanGain / (abs(bestFit) + 1e-8));
    g8 = signedLog(prevMeanStep);

    popMeanNorm = 2 * (popMean - lowerU) ./ rangeU - 1;
    popMeanNorm = min(max(popMeanNorm,-1),1);

    popScaleNorm = log1p(popStd ./ (rangeU + 1e-12));

    state = [g1,g2,g3,g4,g5,g6,g7,g8, ...
        popMeanNorm,popScaleNorm];
end

function y = signedLog(x)
    y = sign(x).*log1p(abs(x));
end
