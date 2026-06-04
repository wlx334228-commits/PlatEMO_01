classdef BLLPCS < ALGORITHM
    % <2026> <multi> <real> <constrained/none> <bilevel>
    % Learning-assisted pool candidate selection for single-objective BLOPs

    methods
        function main(Algorithm,Problem)
            [candidateFactor,warmupGens,minSampleFactor,modelRatio,divRatio] = ...
                Algorithm.ParameterSet(3,2,3,0.7,0.2);

            candidateFactor = max(1,round(candidateFactor));
            warmupGens      = max(0,round(warmupGens));
            minSampleFactor = max(1,round(minSampleFactor));
            modelRatio      = min(max(modelRatio,0),1);
            divRatio        = min(max(divRatio,0),1-modelRatio);
            randRatio       = max(0,1-modelRatio-divRatio);

            totalFElower = 0;
            gen = 1;
            upperTol = 1e-5;

            ulLower = Problem.lower(1:Problem.DU);
            ulUpper = Problem.upper(1:Problem.DU);

            %% Initial upper population and lower responses
            ulPopDec = unifrnd(repmat(ulLower,Problem.N,1),repmat(ulUpper,Problem.N,1));
            llPopDec = zeros(Problem.N,Problem.DL);
            for i = 1 : Problem.N
                [llPopDec(i,:),totalFElower] = llSearch(Problem,ulPopDec(i,:),[],totalFElower);
            end
            Population = Problem.Evaluation([ulPopDec,llPopDec]);

            archiveX = Population.decs;
            archiveX = archiveX(:,1:Problem.DU);
            archiveY = CalFitness(Problem.C,Population);
            finalStats = BLLPCSFinalStats(Problem.FE,totalFElower);

            [~,problemName,runID] = PrepareBLLPCSLogRun(Problem);
            finalCleaner = onCleanup(@()FinalizeBLLPCSRun(problemName,runID,finalStats)); %#ok<NASGU>

            %% Upper-level evolution
            while Algorithm.NotTerminated(Population)
                currentBest = min(CalFitness(Problem.C,Population));
                if abs(currentBest) <= upperTol
                    finalStats.UpperFE = Problem.FE;
                    finalStats.TotalLowerFE = totalFElower;
                    fprintf('Upper optimum reached: BestUpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        currentBest,Problem.FE,totalFElower);
                    break;
                end

                Fitness = CalFitness(Problem.C,Population);
                useModel = gen > warmupGens && size(archiveX,1) >= minSampleFactor * Problem.N;

                if useModel
                    poolSize = candidateFactor * Problem.N;
                else
                    poolSize = Problem.N;
                end

                %% Generate upper candidate pool only by SBX and polynomial mutation
                MatingPool = TournamentSelection(2,poolSize,Fitness);
                ParentDec  = Population(MatingPool).decs;
                candidateUL = OperatorSBXPM(ParentDec(:,1:Problem.DU),ulLower,ulUpper);

                modeName = 'EA';
                selectedLocal = (1:Problem.N)';
                if useModel
                    model = TrainCandidateModel(Problem,archiveX,archiveY,Population);
                    if ~isempty(model)
                        candidateFeature = BuildCandidateFeatures(Problem,candidateUL,Population);
                        predictedFit = PredictCandidateFitness(model,candidateFeature);
                        selectedLocal = SelectCandidatesByModel( ...
                            candidateUL,predictedFit,Problem.N,modelRatio,divRatio,randRatio);
                        modeName = 'LPCS';
                    end
                end

                ulOffDec = candidateUL(selectedLocal,:);

                %% Lower-level DE search with nearest-population warm start
                PopDec = Population.decs;
                PopUL  = PopDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(single(ulOffDec),single(PopUL)),[],2);

                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    warmLL = PopDec(closest(i),Problem.DU+1:end);
                    [llOffDec(i,:),totalFElower] = llSearch(Problem,ulOffDec(i,:),warmLL,totalFElower);
                end

                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);
                Population = EnvironmentalSelectionUpper(Problem,Population,Offspring);

                %% Update online archive with all truly evaluated upper candidates
                offFit = CalFitness(Problem.C,Offspring);
                archiveX = [archiveX;ulOffDec]; %#ok<AGROW>
                archiveY = [archiveY;offFit]; %#ok<AGROW>
                [archiveX,archiveY] = TruncateArchive(archiveX,archiveY,50*Problem.N);

                bestFit = min(CalFitness(Problem.C,Population));
                finalStats.UpperFE = Problem.FE;
                finalStats.TotalLowerFE = totalFElower;
                fprintf('BLLPCS Gen=%4d | Mode=%s | Pool=%4d | UpperFE=%6d | TotalLowerFE=%10d | BestUpperFit=%.6e\n', ...
                    gen,modeName,poolSize,Problem.FE,totalFElower,bestFit);
                drawnow;

                if abs(bestFit) <= upperTol
                    fprintf('Upper optimum reached: BestUpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        bestFit,Problem.FE,totalFElower);
                    break;
                end

                gen = gen + 1;
            end

        end
    end
end
