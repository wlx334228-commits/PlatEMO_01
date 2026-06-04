function value = CriticForward(Critic, state)
% CriticForward
% Forward pass of Critic network
%
% Input:
%   Critic : critic struct from InitializeCriticNetwork
%   state  : 1 x stateDim numeric vector
%
% Output:
%   value  : scalar state value V(s)

    % ===== 1. state -> dlarray =====
    dlState = dlarray(single(state(:)), "CB");

    % ===== 2. forward critic network =====
    dlValue = forward(Critic.net, dlState);

    % ===== 3. extract numeric scalar =====
    value = extractdata(dlValue);
    value = double(value);
end