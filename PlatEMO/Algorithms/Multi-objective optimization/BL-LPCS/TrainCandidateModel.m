function model = TrainCandidateModel(Problem,archiveX,archiveY,Population)
% Train a bagged-tree regression model for candidate ranking.

    model = [];
    valid = all(isfinite(archiveX),2) & isfinite(archiveY);
    archiveX = archiveX(valid,:);
    archiveY = archiveY(valid,:);
    if size(archiveX,1) < 5 || numel(unique(archiveY)) < 2 || exist('fitrensemble','file') ~= 2
        return;
    end

    X = BuildCandidateFeatures(Problem,archiveX,Population);
    y = archiveY(:);

    try
        leafSize = max(1,round(size(X,1)/20));
        if exist('templateTree','file') == 2
            tree = templateTree('MinLeafSize',leafSize);
            model = fitrensemble(X,y, ...
                'Method','Bag', ...
                'NumLearningCycles',80, ...
                'Learners',tree);
        else
            model = fitrensemble(X,y,'Method','Bag','NumLearningCycles',80);
        end
    catch
        model = [];
    end
end
