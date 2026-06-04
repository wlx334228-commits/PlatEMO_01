function merged = AppendPPOBuffer(base, incoming)
% AppendPPOBuffer
% Concatenate one generation of rollout data into the accumulated buffer.

    merged = base;
    fields = fieldnames(incoming);

    for i = 1:numel(fields)
        name = fields{i};
        if isempty(merged.(name))
            merged.(name) = incoming.(name);
        elseif ~isempty(incoming.(name))
            merged.(name) = [merged.(name); incoming.(name)];
        end
    end
end
