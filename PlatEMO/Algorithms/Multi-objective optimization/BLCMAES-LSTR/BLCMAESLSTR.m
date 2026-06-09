classdef BLCMAESLSTR < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    % PlatEMO implementation of BL-CMA-ES with local successful transition reuse.
    methods
        function main(Algorithm,Problem)
            %% Parameters
            [uTol,lTol,uImprFEs,lImprFEs,includeLLConInUpper,archiveScale, ...
                neighborK,reuseThreshold,directionNoise,maxStepRatio] = ...
                Algorithm.ParameterSet(1e-6,1e-6,50,50,0,10,5, ...
                0.25*sqrt(Problem.DU),0.20,0.20);

            BI = BuildBI(Problem,uTol,lTol,uImprFEs,lImprFEs,includeLLConInUpper);
            CMA = InitCMAES(BI);
            Archive = InitializeTransitionArchive(Problem,max(1,round(archiveScale*Problem.N)));
            lstrParams.neighborK = neighborK;
            lstrParams.reuseThreshold = reuseThreshold;
            lstrParams.directionNoise = directionNoise;
            lstrParams.maxStepRatio = maxStepRatio;

            maxIter = ceil(BI.UmaxFEs/CMA.lambda);
            imprIter = max(1,ceil(BI.UmaxImprFEs/CMA.lambda));

            elite = [];
            previousPopulation = [];
            recordUF = [];
            totalFElower = 0;
            upperReached = false;
            runRecordWritten = false;
            gen = 0;

            try
                for iter = 1 : maxIter
                    gen = iter;
                    remainingUpperFE = BI.UmaxFEs - Problem.FE;
                    if remainingUpperFE < 2
                        break;
                    end

                    %% Sampling and LSTR correction
                    popN = min(CMA.lambda,floor(remainingUpperFE/2));
                    sampledDec = zeros(popN,BI.dim);
                    ulBaseDec = zeros(popN,BI.u_dim);
                    for i = 1 : popN
                        sampledDec(i,:) = SampleFullVector(CMA,BI);
                        ulBaseDec(i,:) = sampledDec(i,1:BI.u_dim);
                    end
                    ulDec = GenerateLSTROffspring(ulBaseDec,Archive,Problem,lstrParams);

                    BasePOP = EmptyIndividual(popN);
                    POP = EmptyIndividual(popN);
                    for i = 1 : popN
                        BasePOP(i).UX = ulBaseDec(i,:);
                        BasePOP(i).LX = sampledDec(i,BI.u_dim+1:end);
                        POP(i).UX = ulDec(i,:);
                        POP(i).LX = sampledDec(i,BI.u_dim+1:end);
                    end

                    %% Probe base offspring without LSTR correction
                    for i = 1 : popN
                        [BasePOP(i).LX,BasePOP(i).LF,BasePOP(i).LC,BasePOP(i).RF,totalFElower] = ...
                            LowerLevelSearch(Problem,BasePOP(i).UX,CMA,BI,totalFElower);
                        [BasePOP(i).UF,BasePOP(i).UC,BasePOP(i).Solution] = ...
                            EvaluateUpper(Problem,BasePOP(i).UX,BasePOP(i).LX,BI);
                        BasePOP(i).UFEs = Problem.FE;
                        BasePOP(i).LFEs = totalFElower;
                    end
                    BasePOP = AssignUpperFitness(BasePOP,BI);

                    %% Evaluate LSTR-corrected offspring used by the algorithm
                    for i = 1 : popN
                        [POP(i).LX,POP(i).LF,POP(i).LC,POP(i).RF,totalFElower] = ...
                            LowerLevelSearch(Problem,POP(i).UX,CMA,BI,totalFElower);
                        [POP(i).UF,POP(i).UC,POP(i).Solution] = ...
                            EvaluateUpper(Problem,POP(i).UX,POP(i).LX,BI);
                        POP(i).UFEs = Problem.FE;
                        POP(i).LFEs = totalFElower;
                    end

                    POP = AssignUpperFitness(POP,BI);
                    lstrStats = CalLSTRProbeStats(ulBaseDec,ulDec,BasePOP,POP);

                    %% Elite preservation and refinement
                    rfIdx = find([POP.RF]);
                    if isempty(rfIdx)
                        rfIdx = 1 : length(POP);
                    end
                    [~,bestLocal] = min([POP(rfIdx).fit]);
                    bestIdx = rfIdx(bestLocal);
                    bestIndv = POP(bestIdx);

                    if UpperLevelComparator(bestIndv,elite,BI) || rand > 0.5
                        [bestIndv,totalFElower] = Refine(Problem,bestIndv,CMA,BI,totalFElower);
                        POP(bestIdx) = bestIndv;
                        if UpperLevelComparator(bestIndv,elite,BI)
                            elite = bestIndv;
                        end
                    elseif ~isempty(elite)
                        [elite,totalFElower] = Refine(Problem,elite,CMA,BI,totalFElower);
                    end

                    POP = AssignUpperFitness(POP,BI);
                    if isempty(elite)
                        [~,bestIdx] = min([POP.fit]);
                        elite = POP(bestIdx);
                    end

                    currentPopulation = [POP.Solution];
                    if ~isempty(previousPopulation)
                        Archive = UpdateTransitionArchive(Problem,Archive,previousPopulation,currentPopulation);
                    end
                    previousPopulation = currentPopulation;

                    elite.UFEs = Problem.FE;
                    elite.LFEs = totalFElower;
                    recordUF(end+1) = elite.UF; %#ok<AGROW>

                    %% Termination check
                    [lowerFit,lowerGap] = CalOneLowerFitness(Problem,elite);
                    targetBest = abs(bestIndv.UF - BI.u_fopt) < BI.u_ftol;
                    targetElite = abs(elite.UF - BI.u_fopt) < BI.u_ftol;
                    reachMaxFEs = Problem.FE >= BI.UmaxFEs;
                    reachFlat = false;
                    if iter > imprIter
                        oldUF = recordUF(iter-imprIter+1);
                        curUF = recordUF(iter);
                        reachFlat = abs(curUF-oldUF)/(abs(recordUF(1))+abs(curUF)+eps) < BI.u_ftol && ...
                            abs(curUF-oldUF) < BI.u_ftol;
                    end
                    upperReached = targetBest || targetElite;

                    fprintf(['BLCMAESLSTR Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | ', ...
                        'UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e | RF=%d | ', ...
                        'Archive=%d | Sigma=%.3e | LSTRBetter=%d/%d | LSTRRate=%.2f%% | ', ...
                        'BaseBestFit=%.6e | LSTRBestFit=%.6e\n'], ...
                        iter,Problem.FE,totalFElower,elite.UF,lowerFit,lowerGap, ...
                        elite.RF,size(Archive.X0,1),CMA.sigma, ...
                        lstrStats.Better,lstrStats.Trial,lstrStats.Rate*100, ...
                        lstrStats.BaseBestFit,lstrStats.LSTRBestFit);

                    nofinish = Algorithm.NotTerminated(currentPopulation);

                    if targetBest
                        elite = bestIndv;
                    end
                    if upperReached || reachMaxFEs || reachFlat || ~nofinish
                        break;
                    end

                    CMA = UpdateCMAESFromPOP(CMA,POP,BI);
                end
            catch err
                if strcmp(err.identifier,'PlatEMO:Termination')
                    writeRunRecord();
                end
                rethrow(err);
            end

            writeRunRecord();

            function writeRunRecord()
                if ~runRecordWritten
                    AppendBLCMAESLSTRRunFERecord(Problem,Problem.FE,totalFElower,gen,upperReached);
                    runRecordWritten = true;
                end
            end
        end
    end
end

function BI = BuildBI(Problem,uTol,lTol,uImprFEs,lImprFEs,includeLLConInUpper)
    BI.dim = Problem.D;
    BI.u_dim = Problem.DU;
    BI.l_dim = Problem.DL;
    BI.xrange = [Problem.lower;Problem.upper];
    BI.u_lb = Problem.lower(1:Problem.DU);
    BI.u_ub = Problem.upper(1:Problem.DU);
    BI.l_lb = Problem.lower(Problem.DU+1:end);
    BI.l_ub = Problem.upper(Problem.DU+1:end);
    BI.UmaxFEs = Problem.maxFE;
    BI.LmaxFEs = Problem.maxFElower;
    BI.UmaxImprFEs = max(1,uImprFEs);
    BI.LmaxImprFEs = max(1,lImprFEs);
    BI.u_ftol = uTol;
    BI.l_ftol = lTol;
    BI.u_fopt = 0;
    BI.upperConN = Problem.C;
    BI.isLowerLevelConstraintsIncludedInUpperLevel = logical(includeLLConInUpper);
end

function POP = EmptyIndividual(N)
    indiv = struct('UX',[],'LX',[],'UF',[],'LF',[],'UC',0,'LC',0, ...
        'RF',false,'fit',[],'UFEs',[],'LFEs',[],'Solution',[]);
    POP = repmat(indiv,1,N);
end

function Stats = CalLSTRProbeStats(ulBaseDec,ulLSTRDec,BasePOP,LSTRPOP)
    baseFit = [BasePOP.fit];
    lstrFit = [LSTRPOP.fit];
    perturbed = any(abs(ulLSTRDec-ulBaseDec) > 1e-12,2)';

    Stats.Trial = sum(perturbed);
    if Stats.Trial > 0
        Stats.Better = sum(lstrFit(perturbed) < baseFit(perturbed));
        Stats.Rate = Stats.Better / Stats.Trial;
    else
        Stats.Better = 0;
        Stats.Rate = 0;
    end

    Stats.BaseBestFit = min(baseFit);
    Stats.LSTRBestFit = min(lstrFit);
end

function U = SampleFullVector(CMA,BI)
    U = CMA.xmean + CMA.sigma * randn(1,BI.dim) .* CMA.D * CMA.B';
    over = U > BI.xrange(2,:);
    U(over) = (CMA.xmean(over) + BI.xrange(2,over))/2;
    under = U < BI.xrange(1,:);
    U(under) = (CMA.xmean(under) + BI.xrange(1,under))/2;
end

function [F,C,Solution] = EvaluateUpper(Problem,UX,LX,BI)
    Solution = Problem.Evaluation([UX,LX]);
    F = Solution.obj(1);
    C = SumConstraintViolation(Solution.con(1:min(BI.upperConN,length(Solution.con))));
end

function [F,C,Solution] = EvaluateLower(Problem,UX,LX,BI)
    Solution = Problem.EvaluationLower([UX,LX]);
    F = Solution.obj(2);
    lowerCon = [];
    if length(Solution.con) > BI.upperConN
        lowerCon = Solution.con(BI.upperConN+1:end);
    end
    C = SumConstraintViolation(lowerCon);
end

function C = SumConstraintViolation(Con)
    if isempty(Con)
        C = 0;
    else
        Con(isnan(Con)) = 0;
        C = sum(max(0,Con));
    end
end

function POP = AssignLowerFitness(POP)
    fit = CombineConstraintWithFitness([POP.LF],[POP.LC]);
    for i = 1 : length(POP)
        POP(i).fit = fit(i);
    end
end

function POP = AssignUpperFitness(POP,BI)
    CV = [POP.UC];
    if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
        CV = CV + [POP.LC];
    end
    fit = CombineConstraintWithFitness([POP.UF],CV);
    for i = 1 : length(POP)
        POP(i).fit = fit(i);
    end
end

function fitness = CombineConstraintWithFitness(obj,cv)
    cv = max(0,cv);
    feasible = cv <= 0;
    fitness = obj;
    if any(~feasible)
        if any(feasible)
            base = max(obj(feasible));
        else
            base = 1e10;
        end
        fitness(~feasible) = base + cv(~feasible);
    end
end

function [Q,totalFElower] = Refine(Problem,P,CMA,BI,totalFElower)
    Q = P;
    [Q.LX,Q.LF,Q.LC,Q.RF,totalFElower] = LowerLevelSearch(Problem,Q.UX,CMA,BI,totalFElower);
    if LowerLevelComparator(Q,P)
        Q.RF = max(Q.RF,P.RF);
        [Q.UF,Q.UC,Q.Solution] = EvaluateUpper(Problem,Q.UX,Q.LX,BI);
    else
        Q = P;
    end
end

function noWorse = UpperLevelComparator(P,Q,BI)
    if isempty(Q)
        noWorse = true;
    else
        tmp = AssignUpperFitness([P,Q],BI);
        noWorse = tmp(1).fit <= tmp(2).fit;
    end
end

function noWorse = LowerLevelComparator(P,Q)
    if isempty(Q)
        noWorse = true;
    else
        tmp = AssignLowerFitness([P,Q]);
        noWorse = tmp(1).fit <= tmp(2).fit;
    end
end

function [bestLX,bestLF,bestLC,bestRF,totalFElower] = LowerLevelSearch(Problem,xu,CMA,BI,totalFElower)
    sigma0 = 1;
    LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
    LCMA.sigma = sigma0;
    LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end) * CMA.sigma^2;
    LCMA.pc = CMA.pc(BI.u_dim+1:end) * CMA.sigma;
    LCMA.ps = zeros(1,BI.l_dim);
    lambda = 4 + floor(3*log(BI.l_dim));
    mu = floor(lambda/2);
    weights = log(mu+1/2) - log(1:mu);
    weights = weights/sum(weights);
    mueff = sum(weights)^2/sum(weights.^2);
    cc = (4+mueff/BI.l_dim) / (BI.l_dim+4 + 2*mueff/BI.l_dim);
    cs = (mueff+2) / (BI.l_dim+mueff+5);
    c1 = 2 / ((BI.l_dim+1.3)^2+mueff);
    cmu = min(1-c1, 2*(mueff-2+1/mueff) / ((BI.l_dim+2)^2+mueff));
    damps = 1 + 2*max(0, sqrt((mueff-1)/(BI.l_dim+1))-1) + cs;
    chiN = BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
    [LCMA.B,LCMA.D] = eig(LCMA.C);
    LCMA.D = sqrt(max(diag(LCMA.D),eps))';
    LCMA = RepairCMA(LCMA);
    LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
    cy = sqrt(BI.l_dim) + 2*BI.l_dim/(BI.l_dim+2);

    bestIndv = [];
    bestRF = false;
    maxIter = ceil(BI.LmaxFEs/lambda);
    imprIter = max(1,ceil(BI.LmaxImprFEs/lambda));
    record = zeros(1,maxIter);

    for iter = 1 : maxIter
        Q = EmptyLowerIndividual(lambda);
        for i = 1 : lambda
            Q(i).LX = LCMA.xmean + LCMA.sigma * randn(1,BI.l_dim) .* LCMA.D * LCMA.B';
            over = Q(i).LX > BI.l_ub;
            Q(i).LX(over) = (LCMA.xmean(over) + BI.l_ub(over))/2;
            under = Q(i).LX < BI.l_lb;
            Q(i).LX(under) = (LCMA.xmean(under) + BI.l_lb(under))/2;
            [Q(i).LF,Q(i).LC] = EvaluateLower(Problem,xu,Q(i).LX,BI);
            totalFElower = totalFElower + 1;
        end

        Q = AssignLowerFitness(Q);
        [~,rank] = sort([Q.fit],'ascend');
        xold = LCMA.xmean;
        X = cat(1,Q.LX);
        Y = bsxfun(@minus,X(rank(1:mu),:),xold) / LCMA.sigma;
        Y = bsxfun(@times,Y,min(1,cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
        deltaXmean = weights * Y;
        LCMA.xmean = LCMA.xmean + deltaXmean * LCMA.sigma;
        Cmu = Y' * diag(weights) * Y;
        LCMA.ps = (1-cs)*LCMA.ps + sqrt(cs*(2-cs)*mueff) * deltaXmean * LCMA.invsqrtC;
        LCMA.pc = (1-cc)*LCMA.pc + sqrt(cc*(2-cc)*mueff) * deltaXmean;
        LCMA.C = (1-c1-cmu) * LCMA.C + c1 * (LCMA.pc'*LCMA.pc) + cmu * Cmu;
        deltaSigma = (cs/damps)*(norm(LCMA.ps)/chiN - 1);
        LCMA.sigma = LCMA.sigma * exp(min(CMA.delta_sigma_max,deltaSigma));
        LCMA.C = triu(LCMA.C) + triu(LCMA.C,1)';
        [LCMA.B,LCMA.D] = eig(LCMA.C);
        LCMA.D = sqrt(max(diag(LCMA.D),eps))';
        LCMA = RepairCMA(LCMA);
        LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';

        if LowerLevelComparator(Q(rank(1)),bestIndv)
            bestIndv = Q(rank(1));
        end
        record(iter) = bestIndv.LF;

        if (iter > imprIter && abs(record(iter)-record(iter-imprIter+1))/(abs(record(1))+abs(record(iter))+eps) < 1e-4) || ...
                (iter > imprIter && abs(record(iter)-record(iter-imprIter+1)) < 10*BI.l_ftol) || ...
                LCMA.sigma/sigma0 < 1e-2 || LCMA.sigma/sigma0 > 1e2
            bestRF = true;
            break;
        end
    end

    bestLX = bestIndv.LX;
    bestLF = bestIndv.LF;
    bestLC = bestIndv.LC;
    bestRF = bestRF;
end

function Q = EmptyLowerIndividual(N)
    indiv = struct('LX',[],'LF',[],'LC',0,'fit',[]);
    Q = repmat(indiv,1,N);
end

function CMA = InitCMAES(BI)
    CMA.lambda = 4 + floor(3*log(BI.dim));
    CMA.sigma = 0.3 * median(BI.xrange(2,:) - BI.xrange(1,:));
    CMA.mu = floor(CMA.lambda/2);
    CMA.weights = log(CMA.mu+1/2) - log(1:CMA.mu);
    CMA.weights = CMA.weights/sum(CMA.weights);
    CMA.mueff = sum(CMA.weights)^2/sum(CMA.weights.^2);
    CMA.cc = (4+CMA.mueff/BI.dim) / (BI.dim+4 + 2*CMA.mueff/BI.dim);
    CMA.cs = (CMA.mueff+2) / (BI.dim+CMA.mueff+5);
    CMA.c1 = 2 / ((BI.dim+1.3)^2+CMA.mueff);
    CMA.cmu = min(1-CMA.c1, 2*(CMA.mueff-2+1/CMA.mueff) / ((BI.dim+2)^2+CMA.mueff));
    CMA.damps = 1 + 2*max(0, sqrt((CMA.mueff-1)/(BI.dim+1))-1) + CMA.cs;
    CMA.chiN = BI.dim^0.5*(1-1/(4*BI.dim)+1/(21*BI.dim^2));
    CMA.pc = zeros(1,BI.dim);
    CMA.ps = zeros(1,BI.dim);
    CMA.B = eye(BI.dim);
    CMA.D = ones(1,BI.dim);
    CMA.C = CMA.B * diag(CMA.D.^2) * CMA.B';
    CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
    CMA.xmean = (BI.xrange(2,:) - BI.xrange(1,:)).*rand(1,BI.dim) + BI.xrange(1,:);
    CMA.cy = sqrt(BI.dim) + 2*BI.dim/(BI.dim+2);
    CMA.delta_sigma_max = 1;
end

function CMA = UpdateCMAESFromPOP(CMA,POP,BI)
    [~,rank] = sort([POP.fit],'ascend');
    useMu = min(CMA.mu,length(rank));
    weights = CMA.weights(1:useMu);
    weights = weights/sum(weights);
    mueff = sum(weights)^2/sum(weights.^2);
    Selected = [POP(rank(1:useMu)).Solution];
    X = Selected.decs;
    xold = CMA.xmean;
    Y = bsxfun(@minus,X,xold) / CMA.sigma;
    Y = bsxfun(@times,Y,min(1,CMA.cy./sqrt(sum((Y*CMA.invsqrtC').^2,2))));
    deltaXmean = weights * Y;
    CMA.xmean = CMA.xmean + deltaXmean * CMA.sigma;
    Cmu = Y' * diag(weights) * Y;
    CMA.ps = (1-CMA.cs)*CMA.ps + sqrt(CMA.cs*(2-CMA.cs)*mueff) * deltaXmean * CMA.invsqrtC;
    CMA.pc = (1-CMA.cc)*CMA.pc + sqrt(CMA.cc*(2-CMA.cc)*mueff) * deltaXmean;
    CMA.C = (1-CMA.c1-CMA.cmu) * CMA.C + CMA.c1 * (CMA.pc'*CMA.pc) + CMA.cmu * Cmu;
    deltaSigma = (CMA.cs/CMA.damps)*(norm(CMA.ps)/CMA.chiN - 1);
    CMA.sigma = CMA.sigma * exp(min(deltaSigma,CMA.delta_sigma_max));
    CMA.C = triu(CMA.C) + triu(CMA.C,1)';
    [CMA.B,CMA.D] = eig(CMA.C);
    CMA.D = sqrt(max(diag(CMA.D),eps))';
    CMA = RepairCMA(CMA);
    CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
end

function Model = RepairCMA(Model)
    dim = length(Model.D);
    if any(Model.D <= 0)
        Model.D(Model.D < 0) = 0;
        tmp = max(Model.D)/1e7;
        if tmp <= 0
            tmp = eps;
        end
        Model.C = Model.C + tmp * eye(dim);
        Model.D = Model.D + tmp * ones(1,dim);
    end
    if max(Model.D) > 1e7 * min(Model.D)
        tmp = max(Model.D)/1e7 - min(Model.D);
        Model.C = Model.C + tmp * eye(dim);
        Model.D = Model.D + tmp * ones(1,dim);
    end
    if Model.sigma > 1e7 * max(Model.D)
        fac = Model.sigma / max(Model.D);
        Model.sigma = Model.sigma / fac;
        Model.D = Model.D * fac;
        Model.pc = Model.pc * fac;
        Model.C = Model.C * fac^2;
    end
end

function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,elite)
    LowerFit = elite.LF;
    lowerGap = CalOneLowerGap(Problem,[elite.UX,elite.LX],elite.LF);
end

function lowerGap = CalOneLowerGap(Problem,Dec,FL)
    if ~isprop(Problem,'p')
        lowerGap = inf;
        return;
    end
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
            FLstar = nan;
    end

    if isnan(FLstar)
        lowerGap = inf;
    else
        lowerGap = abs(FL - FLstar);
    end
end
