function [bestLL,totalFElower] = llSearch(Problem,ulDec,warmLL,totalFElower)
% Lower-level search by DE/rand/1/bin for a fixed upper-level solution.

    F  = 0.5;
    CR = 0.9;
    lowerLL = Problem.lower(Problem.DU+1:end);
    upperLL = Problem.upper(Problem.DU+1:end);
    D = Problem.DL;
    N = max(4,Problem.N);

    if nargin < 3 || isempty(warmLL)
        warmLL = zeros(0,D);
    end
    warmLL = warmLL(:,1:D);
    warmLL = min(max(warmLL,repmat(lowerLL,size(warmLL,1),1)),repmat(upperLL,size(warmLL,1),1));
    warmLL = warmLL(1:min(size(warmLL,1),N),:);

    randN = N - size(warmLL,1);
    llDec = [warmLL;unifrnd(repmat(lowerLL,randN,1),repmat(upperLL,randN,1))];

    llPopulation = Problem.EvaluationLower([repmat(ulDec,N,1),llDec]);
    FElower = length(llPopulation);
    totalFElower = totalFElower + length(llPopulation);

    fitness = CalFitness(Problem.C,llPopulation);
    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
    bestLowerStopFit = min(lowerStopFit);
    lowerTol = 1e-5;

    while FElower < Problem.maxFElower && bestLowerStopFit > lowerTol
        trial = zeros(N,D);
        for i = 1 : N
            idx = randperm(N,3);
            while any(idx == i)
                idx = randperm(N,3);
            end
            mutant = llDec(idx(1),:) + F .* (llDec(idx(2),:) - llDec(idx(3),:));
            mutant = min(max(mutant,lowerLL),upperLL);

            trial(i,:) = llDec(i,:);
            jrand = randi(D);
            cross = rand(1,D) <= CR;
            cross(jrand) = true;
            trial(i,cross) = mutant(cross);
        end
        trial = min(max(trial,repmat(lowerLL,N,1)),repmat(upperLL,N,1));

        remain = Problem.maxFElower - FElower;
        evalN = min(N,remain);
        trialPop = Problem.EvaluationLower([repmat(ulDec,evalN,1),trial(1:evalN,:)]);
        FElower = FElower + length(trialPop);
        totalFElower = totalFElower + length(trialPop);

        trialFitness = CalFitness(Problem.C,trialPop);
        replace = trialFitness < fitness(1:evalN);

        if any(replace)
            replaceIndex = find(replace);
            llDec(replaceIndex,:) = trial(replaceIndex,:);
            llPopulation(replaceIndex) = trialPop(replaceIndex);
            fitness(replaceIndex) = trialFitness(replaceIndex);
        end

        lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
        bestLowerStopFit = min(lowerStopFit);
    end

    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
    if all(isinf(lowerStopFit))
        [~,best] = min(fitness);
    else
        [~,best] = min(lowerStopFit);
    end
    bestLL = llPopulation(best).dec(Problem.DU+1:end);
end

function lowerStopFit = CalLowerStopFitness(Problem,llPopulation)
    PopDec = llPopulation.decs;
    PopObj = llPopulation.objs;
    PopCon = llPopulation.cons;
    N = size(PopDec,1);

    lowerGap = nan(N,1);
    for i = 1 : N
        lowerGap(i) = CalLowerGap(Problem,PopDec(i,:),PopObj(i,:));
    end
    lowerGap(isnan(lowerGap)) = inf;

    if isempty(PopCon)
        LLCV = zeros(N,1);
    else
        if size(PopCon,2) >= Problem.C + 1
            llCon = PopCon(:,Problem.C+1:end);
        else
            llCon = [];
        end
        if isempty(llCon)
            LLCV = zeros(N,1);
        else
            LLCV = sum(max(0,llCon),2);
        end
    end

    feasible = LLCV <= 0;
    lowerStopFit = feasible.*lowerGap + ~feasible.*(LLCV + 1e10);
end

function lowerGap = CalLowerGap(Problem,Dec,Obj)
    lowerGap = nan;
    if length(Obj) < 2 || ~isprop(Problem,'p')
        return;
    end

    FL = Obj(2);
    xu1 = Dec(1:Problem.p);
    problemName = class(Problem);

    switch problemName
        case {'SMD1','SMD2','SMD3','SMD4','SMD5','SMD6'}
            FLstar = sum(xu1.^2,2);
        case 'SMD7'
            FLstar = sum(xu1.^3,2);
        case 'SMD8'
            FLstar = sum(abs(xu1),2);
        case {'SMD9','SMD10'}
            FLstar = sum(xu1.^2,2);
        case {'SMD11','SMD12'}
            FLstar = sum(xu1.^2,2) + 1;
        otherwise
            return;
    end
    lowerGap = abs(FL - FLstar);
end
