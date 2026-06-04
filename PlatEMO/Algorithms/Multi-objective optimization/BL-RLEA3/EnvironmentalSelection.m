function Population = EnvironmentalSelection(Problem,Population,Offspring)
% The environmental selection of BL-RLEA2 lower-level search

    selected = randperm(length(Population),2);
    Pool     = [Population(selected),Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    Population(selected) = Pool(rank(1:length(selected)));
end
