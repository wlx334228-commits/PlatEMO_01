function OffDec = OperatorSBXPM(Parent,Lower,Upper)
% Simulated binary crossover and polynomial mutation.

    [proC,disC,proM,disM] = deal(1,20,1,20);

    [N,D] = size(Parent);
    originalN = N;

    Lower = Lower(:)';
    Upper = Upper(:)';
    Span  = Upper - Lower;
    fixed = Span <= 1e-12;
    safeSpan = Span;
    safeSpan(fixed) = 1;

    Parent = Parent(randperm(N),:);
    if mod(N,2) == 1
        Parent = [Parent;Parent(randi(N),:)];
        N = N + 1;
    end

    Parent1 = Parent(1:2:N,:);
    Parent2 = Parent(2:2:N,:);
    Npair   = size(Parent1,1);

    beta = zeros(Npair,D);
    mu   = rand(Npair,D);
    beta(mu<=0.5) = (2.*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2.*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta .* (-1).^randi([0,1],Npair,D);
    beta(rand(Npair,D)<0.5) = 1;
    beta(repmat(rand(Npair,1)>proC,1,D)) = 1;

    Off1 = (Parent1+Parent2)/2 + beta.*(Parent1-Parent2)/2;
    Off2 = (Parent1+Parent2)/2 - beta.*(Parent1-Parent2)/2;

    OffDec = zeros(2*Npair,D);
    OffDec(1:2:end,:) = Off1;
    OffDec(2:2:end,:) = Off2;
    if mod(originalN,2) == 1
        if rand < 0.5
            OffDec(end-1,:) = [];
        else
            OffDec(end,:) = [];
        end
    end
    OffDec = OffDec(1:originalN,:);

    LowerM = repmat(Lower,originalN,1);
    UpperM = repmat(Upper,originalN,1);
    SpanM  = repmat(safeSpan,originalN,1);

    OffDec = min(max(OffDec,LowerM),UpperM);
    OffDec(:,fixed) = LowerM(:,fixed);

    Site = rand(originalN,D) < proM/D;
    Site(:,fixed) = false;
    mu = rand(originalN,D);

    temp = Site & mu <= 0.5;
    OffDec(temp) = OffDec(temp) + SpanM(temp).* ...
        ((2.*mu(temp)+(1-2.*mu(temp)).* ...
        (1-(OffDec(temp)-LowerM(temp))./SpanM(temp)).^(disM+1)).^(1/(disM+1))-1);

    temp = Site & mu > 0.5;
    OffDec(temp) = OffDec(temp) + SpanM(temp).* ...
        (1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).* ...
        (1-(UpperM(temp)-OffDec(temp))./SpanM(temp)).^(disM+1)).^(1/(disM+1)));

    OffDec = min(max(OffDec,LowerM),UpperM);
    OffDec(:,fixed) = LowerM(:,fixed);
end
