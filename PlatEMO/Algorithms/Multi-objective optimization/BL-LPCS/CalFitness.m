function Fitness = CalFitness(C,Population)
% Constraint-handled scalar fitness for upper or lower level.

    PopObj = Population.objs;
    PopCon = Population.cons;

    if any(isnan(PopObj(:,1)))
        PopObj = PopObj(:,2);
        if isempty(PopCon)
            PopCon = [];
        else
            PopCon = PopCon(:,C+1:end);
        end
    else
        PopObj = PopObj(:,1);
        if isempty(PopCon)
            PopCon = [];
        else
            PopCon = PopCon(:,1:C);
        end
    end

    if isempty(PopCon)
        CV = zeros(size(PopObj,1),1);
    else
        CV = sum(max(0,PopCon),2);
    end

    feasible = CV <= 0;
    Fitness = feasible.*PopObj + ~feasible.*(CV + 1e10);
end
