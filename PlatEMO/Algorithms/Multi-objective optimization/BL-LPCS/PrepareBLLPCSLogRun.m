function [logDir,problemName,runID] = PrepareBLLPCSLogRun(Problem)
% Prepare the diagnostic output folder for BLLPCS.

    logDir = fileparts(mfilename('fullpath'));
    problemName = class(Problem);
    runID = sprintf('%s_%06d',datestr(now,'yyyymmdd_HHMMSS'),randi(1e6));
end
