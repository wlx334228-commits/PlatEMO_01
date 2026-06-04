function Population = EnvironmentalSelectionUpper(Problem,Population,Offspring)
% Select the best N upper-level individuals by scalar constrained fitness.

    N = length(Population);
    Pool = [Population,Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    Population = Pool(rank(1:N));
end
