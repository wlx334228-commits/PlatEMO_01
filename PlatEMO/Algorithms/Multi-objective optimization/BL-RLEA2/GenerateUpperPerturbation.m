function deltaUL = GenerateUpperPerturbation(PerturbTheta,N,DU)
% Sample additive upper-level perturbations from the PPO-controlled policy.

    if N <= 0
        deltaUL = zeros(0,DU);
        return;
    end

    deltaUL = repmat(PerturbTheta.mu,N,1) + ...
        randn(N,DU) .* repmat(PerturbTheta.sigma,N,1);
end
