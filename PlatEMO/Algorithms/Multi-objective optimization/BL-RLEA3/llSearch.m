function [eliteIndiv,totalFElower] = llSearch(Problem,ulPopDec,llPopDec,totalFElower)
% Obtain the lower-level member corresponding to one upper-level member.

    %% Lower level population initialization
    llPopDec = [llPopDec;unifrnd( ...
        repmat(Problem.lower(Problem.DU+1:end),Problem.N-size(llPopDec,1),1), ...
        repmat(Problem.upper(Problem.DU+1:end),Problem.N-size(llPopDec,1),1))];

    %% Lower-level evaluation by PlatEMO
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,Problem.N,1),llPopDec]);

    FElower = length(llPopulation);
    totalFElower = totalFElower + length(llPopulation);
    lowerTol = 1e-5;

    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
    bestLowerStopFit = min(lowerStopFit);

    %% Lower-level optimization
    while LowerNotTerminated(FElower,Problem.maxFElower,bestLowerStopFit,lowerTol)
        MatingPool = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
        ParentDec  = llPopulation(MatingPool).decs;

        llOffDec = OperatorPCX( ...
            ParentDec(:,Problem.DU+1:end), ...
            Problem.lower(Problem.DU+1:end), ...
            Problem.upper(Problem.DU+1:end));

        llOffspring = Problem.EvaluationLower( ...
            [repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);

        FElower = FElower + length(llOffspring);
        totalFElower = totalFElower + length(llOffspring);

        llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);

        lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
        bestLowerStopFit = min(lowerStopFit);
    end

    %% Return best lower-level solution
    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);

    if all(isinf(lowerStopFit))
        [~,best] = min(CalFitness(Problem.C,llPopulation));
    else
        [~,best] = min(lowerStopFit);
    end

    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
end

function nofinish = LowerNotTerminated(FElower,maxFElower,bestLowerStopFit,lowerTol)
    nofinish = (FElower < maxFElower) && (bestLowerStopFit > lowerTol);
end

function lowerStopFit = CalLowerStopFitness(Problem,llPopulation)
    PopDec = llPopulation.decs;
    PopObj = llPopulation.objs;
    PopCon = llPopulation.cons;

    N = size(PopDec,1);
    lowerGap = nan(N,1);

    for i = 1:N
        lowerGap(i) = CalLowerGap(Problem,PopDec(i,:),PopObj(i,:));
    end

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

    lowerGap(isnan(lowerGap)) = inf;

    Feasible = LLCV <= 0;
    lowerStopFit = Feasible.*lowerGap + ~Feasible.*(LLCV + 1e10);
end

function lowerGap = CalLowerGap(Problem,Dec,Obj)
    problemName = class(Problem);

    if length(Obj) < 2
        lowerGap = nan;
        return;
    end

    FL = Obj(2);

    if ~isprop(Problem,'p') || ~isprop(Problem,'r') || ~isprop(Problem,'q')
        lowerGap = nan;
        return;
    end

    xu1 = Dec(1:Problem.p);

    switch problemName
        case {'SMD1','SMD2','SMD3','SMD4','SMD5','SMD6'}
            FLstar = sum(xu1.^2,2);
        case 'SMD7'
            FLstar = sum(xu1.^3,2);
        case 'SMD8'
            FLstar = sum(abs(xu1),2);
        case {'SMD9','SMD10'}
            FLstar = sum(xu1.^2,2);
        case 'SMD11'
            FLstar = sum(xu1.^2,2) + 1;
        case 'SMD12'
            FLstar = sum(xu1.^2,2) + 1;
        otherwise
            lowerGap = nan;
            return;
    end

    lowerGap = abs(FL - FLstar);
end
