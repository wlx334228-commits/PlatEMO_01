function predictedFit = PredictCandidateFitness(model,Feature)
% Predict upper-level scalar fitness; smaller values are preferred.

    try
        predictedFit = predict(model,Feature);
    catch
        predictedFit = inf(size(Feature,1),1);
    end
    predictedFit = predictedFit(:);
    predictedFit(~isfinite(predictedFit)) = inf;
end
