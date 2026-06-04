classdef BLLPCSFinalStats < handle
% Mutable final counters used by the PlatEMO termination cleanup.

    properties
        UpperFE
        TotalLowerFE
    end

    methods
        function obj = BLLPCSFinalStats(upperFE,totalLowerFE)
            obj.UpperFE = upperFE;
            obj.TotalLowerFE = totalLowerFE;
        end
    end
end
