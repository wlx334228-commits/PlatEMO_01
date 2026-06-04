function [Population,ppoSurvivalRate,ppoSurvived] = EnvironmentalSelectionUpper(Problem,Population,Offspring,Nppo,ppoOffset)
% Generational environmental selection for upper-level population

    N = length(Population);
    Pool = [Population,Offspring];

    [~,rank] = sort(CalFitness(Problem.C,Pool));

    selected = rank(1:N);
    Population = Pool(selected);

    if nargin < 4 || isempty(Nppo) || Nppo <= 0
        ppoSurvivalRate = 0;
        ppoSurvived = false(0,1);
    else
        if nargin < 5 || isempty(ppoOffset)
            ppoOffset = 1;
        end

        ppoStart = N + ppoOffset;
        ppoEnd   = ppoStart + Nppo - 1;
        ppoSurvived = false(Nppo,1);
        survivedIndex = selected(selected >= ppoStart & selected <= ppoEnd) - ppoStart + 1;
        ppoSurvived(survivedIndex) = true;
        ppoSurvivalRate = mean(ppoSurvived);
    end
end
