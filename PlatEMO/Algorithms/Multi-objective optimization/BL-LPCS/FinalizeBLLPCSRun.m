function FinalizeBLLPCSRun(problemName,runID,finalStats)
% Write one final FE record when PlatEMO terminates the run.

    try
        AppendRunFERecord(problemName,runID,finalStats.UpperFE,finalStats.TotalLowerFE);
    catch err
        warning('BLLPCS:FinalizeFailed','Failed to write final FE record: %s',err.message);
    end
end
