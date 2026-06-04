classdef BLRLEA3 < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    methods
        function main(Algorithm,Problem)
            %% PPO parameters
            sigma0 = 1;
            stateDim = 8 + 2 * Problem.DU;
            actionDim = 2 * Problem.DU;
            hiddenDim = 32;
            learnRate = 1e-3;
            enablePerturbDiagnostic = true;

            rolloutUpdateGap = 3;
            rolloutGenCount  = 0;
            rolloutBuffer    = InitializePPOBuffer();

            Actor  = InitializeActorNetwork(stateDim,actionDim,hiddenDim,learnRate);
            Critic = InitializeCriticNetwork(stateDim,hiddenDim,learnRate);

            %% Total lower-level function evaluations
            totalFElower = 0;

            %% Generate random upper population
            ulPopDec = unifrnd( ...
                repmat(Problem.lower(1:Problem.DU),Problem.N,1), ...
                repmat(Problem.upper(1:Problem.DU),Problem.N,1));

            llPopDec = zeros(Problem.N,Problem.DL);
            for i = 1 : size(ulPopDec,1)
                [llPopDec(i,:),totalFElower] = llSearch(Problem,ulPopDec(i,:),[],totalFElower);
            end

            Population = Problem.Evaluation([ulPopDec,llPopDec]);

            % State memory for the next PPO decision
            stateOldBest = min(CalFitness(Problem.C,Population));
            noImproveGen = 0;
            prevPPOBetterRate = 0;
            prevMeanSigma = 0;
            prevMeanGain = 0;
            prevMeanStep = 0;
            upperTol = 1e-5;
            upperReached = false;
            UpperFit = stateOldBest;
            LowerFit = nan;
            lowerGap = nan;

            %% Debug logs
            gen = 1;
            [logDir,problemName,problemFileName,runID] = PrepareBLRLEA3LogRun(Problem);
            debugFile = fullfile(logDir,sprintf('BLRLEA3_debug_%s_%s.txt',problemFileName,runID));

            fid = OpenWriteLog(debugFile);
            logCleaner = onCleanup(@()CloseLogFile(fid));
            fprintf(fid,'Problem\tRunID\tGen\tUpperFE\tTotalLowerFE\tUpperFit\tLowerFit\tlowerGap\n');

            diagHistory = zeros(0,10);
            if enablePerturbDiagnostic
                diagFile = fullfile(logDir,sprintf('BLRLEA3_ppo_perturb_diagnostic_%s_%s.tsv',problemFileName,runID));
                diagAllFile = fullfile(logDir,'BLRLEA3_ppo_perturb_diagnostic_all.tsv');
                summaryFile = fullfile(logDir,'BLRLEA3_ppo_perturb_summary.tsv');

                diagHeader = PerturbDiagnosticHeader();
                diagFid = OpenWriteLog(diagFile);
                diagAllFid = OpenAppendLogWithHeader(diagAllFile,diagHeader);

                diagCleaner = onCleanup(@()CloseLogFile(diagFid));
                diagAllCleaner = onCleanup(@()CloseLogFile(diagAllFid));
                fprintf(diagFid,'%s\n',diagHeader);
            end

            %% UL Optimization
            while Algorithm.NotTerminated(Population)
                if upperReached
                    break;
                end

                OldPopulation = Population;

                %% 1. Build global PPO state
                globalState = BuildState(Problem,Population,stateOldBest,noImproveGen, ...
                    prevPPOBetterRate,prevMeanSigma,prevMeanGain,prevMeanStep);

                %% 2. Generate a full EA offspring set and perturb all of it by PPO
                Fitness = CalFitness(Problem.C,Population);
                Noff = Problem.N;

                MatingPoolBase = TournamentSelection(2,Noff,Fitness);
                ParentDecBase  = Population(MatingPoolBase).decs;
                baseEA = OperatorSBXPM( ...
                    ParentDecBase(:,1:Problem.DU), ...
                    Problem.lower(1:Problem.DU), ...
                    Problem.upper(1:Problem.DU));

                valuePPO = CriticForward(Critic,globalState);
                [actionPPO,logProbPPO,actionMeanPPO,actionStdPPO] = ...
                    ActorForward(Actor,globalState);

                % The actor controls a perturbation distribution.  Each PPO
                % sample is paired with the same unperturbed EA offspring.
                PerturbTheta = ActionToParam(actionPPO,Problem,Population,sigma0);
                perturbScale = BuildPopulationPerturbScale(Problem,Population);
                deltaPPO = GenerateUpperPerturbation(PerturbTheta,Noff,Problem.DU);
                ulPPO = baseEA + deltaPPO;

                lowerUL = repmat(Problem.lower(1:Problem.DU),Noff,1);
                upperUL = repmat(Problem.upper(1:Problem.DU),Noff,1);
                ulPPO = min(max(ulPPO,lowerUL),upperUL);

                % Reward slicing relies on this order.  Only ulPPO enters
                % environmental selection; baseEA is evaluated as the
                % unperturbed counterfactual baseline for PPO credit.
                pairedOffDec = [baseEA;ulPPO];

                %% 3. Warm-start lower-level search for paired counterfactual offspring
                AllDec = Population.decs;
                AllUL  = AllDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(single(pairedOffDec),single(AllUL)),[],2);

                llOffDec = zeros(size(pairedOffDec,1),Problem.DL);
                for i = 1 : size(pairedOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearch( ...
                        Problem, ...
                        pairedOffDec(i,:), ...
                        AllDec(closest(i),Problem.DU+1:end), ...
                        totalFElower);
                end

                %% 4. Evaluate paired offspring
                PairedOffspring = Problem.Evaluation([pairedOffDec,llOffDec]);
                BaseOffspring = PairedOffspring(1:Noff);
                Offspring = PairedOffspring(Noff+1:2*Noff);

                %% 5. Optional diagnostic: compare anchors before and after PPO perturbation
                if enablePerturbDiagnostic
                    diagFElower = 0;
                    diagStats = CalPerturbDiagnostic( ...
                        Problem,BaseOffspring,Offspring, ...
                        deltaPPO,perturbScale);

                    diagRow = [gen,diagStats.improveRate,diagStats.meanGain, ...
                        diagStats.medianGain,diagStats.baseMeanFit,diagStats.ppoMeanFit, ...
                        diagStats.baseBestFit,diagStats.ppoBestFit,diagStats.meanStep,diagFElower];
                    diagHistory = [diagHistory;diagRow]; %#ok<AGROW>

                    WritePerturbDiagnosticRow(diagFid,problemName,runID,diagRow);
                    WritePerturbDiagnosticRow(diagAllFid,problemName,runID,diagRow);

                    fprintf('PPODiag Gen=%4d | ImproveRate=%.3f | MeanGain=%.3e | BaseBest=%.3e | PPOBest=%.3e | MeanStep=%.3e\n', ...
                        gen,diagStats.improveRate,diagStats.meanGain,diagStats.baseBestFit, ...
                        diagStats.ppoBestFit,diagStats.meanStep);
                end

                %% 6. Environmental selection
                [Population,ppoSurvivalRate,ppoSurvived] = EnvironmentalSelectionUpper( ...
                    Problem,Population,Offspring,Noff,1);

                %% 7. Generation-level PPO reward after environmental selection
                [rewardPPO,rewardStats] = RewardCalculator( ...
                    Problem,[BaseOffspring,Offspring],Noff, ...
                    ppoSurvived,deltaPPO,perturbScale);

                %% 8. Record best upper and lower information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;

                [LowerFit,lowerGap] = CalOneLowerFitness(Problem,bestDec,bestObj,bestCon);

                fprintf('Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e\n', ...
                    gen, Problem.FE, totalFElower, UpperFit, LowerFit, lowerGap);

                fprintf(fid,'%s\t%s\t%d\t%d\t%d\t%.12e\t%.12e\t%.12e\n', ...
                    problemName,runID,gen,Problem.FE,totalFElower,UpperFit,LowerFit,lowerGap);

                drawnow;

                %% 9. Build generation statistics
                oldBest = min(CalFitness(Problem.C,OldPopulation));
                meanSigma = mean(PerturbTheta.sigma);

                if UpperFit < oldBest - 1e-12
                    nextNoImproveGen = 0;
                else
                    nextNoImproveGen = noImproveGen + 1;
                end

                nextGlobalState = BuildState(Problem,Population,oldBest,nextNoImproveGen, ...
                    rewardStats.betterRate,meanSigma, ...
                    rewardStats.meanGain,rewardStats.meanStep);

                %% 10. Save one multi-step transition per generation
                oneBuffer = InitializePPOBuffer();
                oneBuffer.states       = globalState;
                oneBuffer.actions      = actionPPO;
                oneBuffer.rewards      = rewardPPO;
                oneBuffer.next_states  = nextGlobalState;
                oneBuffer.log_probs    = logProbPPO;
                oneBuffer.values       = valuePPO;
                oneBuffer.action_means = actionMeanPPO;
                oneBuffer.action_stds  = actionStdPPO;

                if abs(UpperFit) <= upperTol
                    upperReached = true;
                    fprintf('Upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit, Problem.FE, totalFElower);
                end

                oneBuffer.dones = double(upperReached);

                rolloutBuffer = AppendPPOBuffer(rolloutBuffer,oneBuffer);
                rolloutGenCount = rolloutGenCount + 1;

                %% 11. PPO update
                if upperReached || rolloutGenCount >= rolloutUpdateGap
                    [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
                    rolloutBuffer = InitializePPOBuffer();
                    rolloutGenCount = 0;
                end

                %% 12. Update state memory for next generation
                gen = gen + 1;
                stateOldBest = oldBest;
                noImproveGen = nextNoImproveGen;
                prevPPOBetterRate = rewardStats.betterRate;
                prevMeanSigma = meanSigma;
                prevMeanGain = rewardStats.meanGain;
                prevMeanStep = rewardStats.meanStep;
            end

            if rolloutGenCount > 0 && ~isempty(rolloutBuffer.states)
                rolloutBuffer.dones(end) = 1;
                [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
            end

            if enablePerturbDiagnostic
                AppendPerturbSummary(summaryFile,problemName,runID,Problem, ...
                    diagHistory,UpperFit,LowerFit,lowerGap,Problem.FE,totalFElower);

                fprintf('PPO diagnostic saved: %s\n',diagFile);
                fprintf('PPO diagnostic all-runs table: %s\n',diagAllFile);
                fprintf('PPO diagnostic summary: %s\n',summaryFile);
            end

            AppendRunFERecord(problemName,runID,Problem.FE,totalFElower);
        end
    end
end

function stats = CalPerturbDiagnostic(Problem,BaseOffspring,PPOOffspring,deltaPPO,perturbScale)
    baseFit = CalFitness(Problem.C,BaseOffspring);
    ppoFit  = CalFitness(Problem.C,PPOOffspring);
    baseFit = baseFit(:);
    ppoFit  = ppoFit(:);

    gain = baseFit - ppoFit;

    perturbScale = perturbScale(:)';
    scaleMat = repmat(perturbScale,size(deltaPPO,1),1);
    stepSize = mean(abs(deltaPPO ./ (scaleMat + 1e-12)),2);

    stats.improveRate = mean(gain > 0);
    stats.meanGain    = mean(gain);
    stats.medianGain  = median(gain);
    stats.baseMeanFit = mean(baseFit);
    stats.ppoMeanFit  = mean(ppoFit);
    stats.baseBestFit = min(baseFit);
    stats.ppoBestFit  = min(ppoFit);
    stats.meanStep    = mean(stepSize);
end

function perturbScale = BuildPopulationPerturbScale(Problem,Population)
    PopDec = Population.decs;
    ulPop = PopDec(:,1:Problem.DU);
    DU = Problem.DU;

    popStd = std(ulPop,0,1);
    popStd(popStd < 1e-6) = 1e-6;

    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    fitness = CalFitness(Problem.C,Population);
    bestAbs = abs(min(fitness));

    if bestAbs > 1e-2
        scale = 0.02;
    elseif bestAbs > 1e-4
        scale = 0.005;
    else
        scale = 1e-5;
    end

    minSigma = max(scale .* range,1e-6);
    perturbScale = max(popStd,minSigma);
end

function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,Dec,Obj,Con)
    if length(Obj) >= 2
        FL = Obj(2);
    else
        FL = nan;
    end

    lowerGap = CalOneLowerGap(Problem,Dec,Obj);

    if isempty(Con) || length(Con) < Problem.C + 1
        LLCV = 0;
    else
        llCon = Con(Problem.C+1:end);
        if isempty(llCon)
            LLCV = 0;
        else
            LLCV = sum(max(0,llCon));
        end
    end

    if LLCV <= 0
        LowerFit = FL;
    else
        LowerFit = LLCV + 1e10;
    end
end

function lowerGap = CalOneLowerGap(Problem,Dec,Obj)
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

function [logDir,problemName,problemFileName,runID] = PrepareBLRLEA3LogRun(Problem)
    persistent runCounter
    if isempty(runCounter)
        runCounter = 0;
    end
    runCounter = runCounter + 1;

    logDir = fullfile(pwd,'BLRLEA3_Diagnostics');
    if ~exist(logDir,'dir')
        mkdir(logDir);
    end

    problemName = class(Problem);
    problemFileName = regexprep(problemName,'[^A-Za-z0-9_]','_');
    if isempty(problemFileName)
        problemFileName = 'Problem';
    end

    t = clock;
    sec = floor(t(6));
    milliSec = floor((t(6) - sec) * 1000);
    runID = sprintf('%04d%02d%02d_%02d%02d%02d_%03d_%03d', ...
        t(1),t(2),t(3),t(4),t(5),sec,milliSec,runCounter);
end

function header = PerturbDiagnosticHeader()
    header = ['Problem\tRunID\tGen\tImproveRate\tMeanGain\tMedianGain\t' ...
        'BaseMeanFit\tPPOMeanFit\tBaseBestFit\tPPOBestFit\t' ...
        'MeanStep\tDiagLowerFE'];
end

function WritePerturbDiagnosticRow(fid,problemName,runID,row)
    fprintf(fid,'%s\t%s\t%d\t%.6f\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%d\n', ...
        problemName,runID,row(1),row(2),row(3),row(4),row(5),row(6), ...
        row(7),row(8),row(9),row(10));
end

function AppendPerturbSummary(summaryFile,problemName,runID,Problem,diagHistory, ...
    finalUpperFit,finalLowerFit,finalLowerGap,upperFE,totalFElower)

    header = ['Problem\tRunID\tDU\tDL\tN\tMaxFE\tGens\t' ...
        'MeanImproveRate\tMeanGain\tMedianGain\tPositiveMeanGainRate\t' ...
        'LastImproveRate\tLastMeanGain\tLastMedianGain\tLastBaseBestFit\tLastPPOBestFit\t' ...
        'FinalUpperFit\tFinalLowerFit\tFinalLowerGap\tUpperFE\tTotalLowerFE\tDiagLowerFETotal'];

    fid = OpenAppendLogWithHeader(summaryFile,header);
    cleaner = onCleanup(@()CloseLogFile(fid)); %#ok<NASGU>

    if isempty(diagHistory)
        gens = 0;
        meanImproveRate = nan;
        meanGain = nan;
        medianGain = nan;
        positiveMeanGainRate = nan;
        lastImproveRate = nan;
        lastMeanGain = nan;
        lastMedianGain = nan;
        lastBaseBestFit = nan;
        lastPPOBestFit = nan;
        diagLowerFETotal = 0;
    else
        gens = size(diagHistory,1);
        meanImproveRate = mean(diagHistory(:,2));
        meanGain = mean(diagHistory(:,3));
        medianGain = median(diagHistory(:,4));
        positiveMeanGainRate = mean(diagHistory(:,3) > 0);
        lastImproveRate = diagHistory(end,2);
        lastMeanGain = diagHistory(end,3);
        lastMedianGain = diagHistory(end,4);
        lastBaseBestFit = diagHistory(end,7);
        lastPPOBestFit = diagHistory(end,8);
        diagLowerFETotal = sum(diagHistory(:,10));
    end

    fprintf(fid,'%s\t%s\t%d\t%d\t%d\t%g\t%d\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%d\t%d\t%d\n', ...
        problemName,runID,Problem.DU,Problem.DL,Problem.N,GetProblemMaxFE(Problem),gens, ...
        meanImproveRate,meanGain,medianGain,positiveMeanGainRate, ...
        lastImproveRate,lastMeanGain,lastMedianGain,lastBaseBestFit,lastPPOBestFit, ...
        finalUpperFit,finalLowerFit,finalLowerGap,upperFE,totalFElower,diagLowerFETotal);
end

function maxFE = GetProblemMaxFE(Problem)
    if isprop(Problem,'maxFE')
        maxFE = Problem.maxFE;
    else
        maxFE = nan;
    end
end

function fid = OpenWriteLog(filePath)
    fid = fopen(filePath,'w');
    if fid < 0
        error('BLRLEA3:LogOpenFailed','Cannot open log file: %s',filePath);
    end
end

function fid = OpenAppendLogWithHeader(filePath,header)
    fileInfo = dir(filePath);
    needHeader = isempty(fileInfo) || fileInfo.bytes == 0;

    fid = fopen(filePath,'a');
    if fid < 0
        error('BLRLEA3:LogOpenFailed','Cannot open log file: %s',filePath);
    end

    if needHeader
        fprintf(fid,'%s\n',header);
    end
end

function AppendRunFERecord(problemName,runID,upperFE,totalFElower)
    algDir = fileparts(mfilename('fullpath'));
    recordFile = fullfile(algDir,'BLRLEA3_run_fe.tsv');
    header = 'Problem\tRunID\tUpperFE\tTotalLowerFE';

    fid = OpenAppendLogWithHeader(recordFile,header);
    cleaner = onCleanup(@()CloseLogFile(fid)); %#ok<NASGU>

    fprintf(fid,'%s\t%s\t%d\t%d\n', ...
        problemName,runID,upperFE,totalFElower);
end

function CloseLogFile(fid)
    if ~isempty(fid) && isnumeric(fid) && fid > 0
        fclose(fid);
    end
end
