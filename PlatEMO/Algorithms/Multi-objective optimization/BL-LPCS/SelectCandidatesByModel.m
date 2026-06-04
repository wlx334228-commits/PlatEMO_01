function selected = SelectCandidatesByModel(CandidateUL,predictedFit,N,modelRatio,divRatio,randRatio)
% Select N candidates by prediction, diversity, and random exploration.

    poolSize = size(CandidateUL,1);
    N = min(N,poolSize);
    selected = zeros(0,1);

    nModel = min(N,round(modelRatio*N));
    nDiv   = min(N-nModel,round(divRatio*N));
    nRand  = N - nModel - nDiv;
    if randRatio <= 0
        nDiv = N - nModel;
        nRand = 0;
    end

    [~,order] = sort(predictedFit,'ascend');
    selected = AddUnique(selected,order,nModel);

    remaining = setdiff((1:poolSize)',selected,'stable');
    for i = 1 : nDiv
        if isempty(remaining)
            break;
        end
        if isempty(selected)
            pick = remaining(randi(length(remaining)));
        else
            dist = pdist2(CandidateUL(remaining,:),CandidateUL(selected,:));
            minDist = min(dist,[],2);
            [~,loc] = max(minDist);
            pick = remaining(loc);
        end
        selected = [selected;pick]; %#ok<AGROW>
        remaining(remaining == pick) = [];
    end

    remaining = setdiff((1:poolSize)',selected,'stable');
    if nRand > 0 && ~isempty(remaining)
        addN = min(nRand,length(remaining));
        selected = [selected;remaining(randperm(length(remaining),addN))]; %#ok<AGROW>
    end

    if length(selected) < N
        remaining = setdiff((1:poolSize)',selected,'stable');
        selected = [selected;remaining(1:N-length(selected))]; %#ok<AGROW>
    end

    selected = selected(1:N);
end

function selected = AddUnique(selected,order,nAdd)
    for i = 1 : length(order)
        if length(selected) >= nAdd
            break;
        end
        if ~ismember(order(i),selected)
            selected = [selected;order(i)]; %#ok<AGROW>
        end
    end
end
