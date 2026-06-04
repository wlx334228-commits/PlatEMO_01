function Fitness = CalFitness(C,Population)
    PopObj = Population.objs;
    PopCon = Population.cons;
    if any(isnan(PopObj(:,1)))  % Lower level
        PopObj = PopObj(:,2);
        PopCon = PopCon(:,C+1:end);
    else                        % Upper level
        PopObj = PopObj(:,1);
        PopCon = PopCon(:,1:C);
    end
    if isempty(PopCon)
        PopCon = zeros(size(PopObj,1),1);
    else
        PopCon = sum(max(0,PopCon),2);
    end
    Feasible = PopCon <= 0;
    Fitness  = Feasible.*PopObj + ~Feasible.*(PopCon+1e10);
end
