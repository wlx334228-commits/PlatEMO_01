function buffer = InitializePPOBuffer()
% InitializePPOBuffer
% Create an empty rollout buffer for PPO trajectory collection.

    buffer.states       = [];
    buffer.actions      = [];
    buffer.rewards      = [];
    buffer.next_states  = [];
    buffer.log_probs    = [];
    buffer.values       = [];
    buffer.action_means = [];
    buffer.action_stds  = [];
    buffer.dones        = [];
end
