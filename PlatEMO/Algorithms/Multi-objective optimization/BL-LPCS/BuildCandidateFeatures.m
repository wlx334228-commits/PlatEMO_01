function Feature = BuildCandidateFeatures(Problem,CandidateUL,Population)
% Build context-aware features for upper-level candidate ranking.

    if isempty(CandidateUL)
        Feature = zeros(0,Problem.DU+7);
        return;
    end

    lower = Problem.lower(1:Problem.DU);
    upper = Problem.upper(1:Problem.DU);
    span = upper - lower;
    span(span <= 1e-12) = 1;

    normUL = (CandidateUL - repmat(lower,size(CandidateUL,1),1)) ./ ...
        repmat(span,size(CandidateUL,1),1);

    PopDec = Population.decs;
    PopUL = PopDec(:,1:Problem.DU);
    Fitness = CalFitness(Problem.C,Population);
    [bestFit,best] = min(Fitness);

    normPop = (PopUL - repmat(lower,size(PopUL,1),1)) ./ repmat(span,size(PopUL,1),1);
    popMean = mean(normPop,1);
    popStd  = std(normPop,0,1);
    bestUL  = normPop(best,:);

    distToBest = sqrt(sum((normUL - repmat(bestUL,size(normUL,1),1)).^2,2)) ./ sqrt(Problem.DU);
    distToMean = sqrt(sum((normUL - repmat(popMean,size(normUL,1),1)).^2,2)) ./ sqrt(Problem.DU);
    nearDist   = min(pdist2(normUL,normPop),[],2) ./ sqrt(Problem.DU);
    popDiv     = mean(popStd);

    fitMean = mean(Fitness);
    fitStd  = std(Fitness);
    fitRange = max(Fitness) - min(Fitness);
    if fitStd <= 1e-12
        fitStd = 1;
    end
    if fitRange <= 1e-12
        fitRange = 1;
    end

    globalInfo = repmat([ ...
        popDiv, ...
        (bestFit - fitMean) / fitStd, ...
        fitStd / (abs(fitMean) + 1e-12), ...
        fitRange / (abs(fitMean) + 1e-12)],size(normUL,1),1);

    Feature = [normUL,distToBest,distToMean,nearDist,globalInfo];
    Feature(~isfinite(Feature)) = 0;
end
