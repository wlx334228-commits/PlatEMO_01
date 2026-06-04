function AppendRunFERecord(problemName,runID,upperFE,totalLowerFE)
% Append one final FE record for each completed problem run.

    logDir = fileparts(mfilename('fullpath'));
    file = fullfile(logDir,'BLLPCS_run_fe.tsv');
    needHeader = ~exist(file,'file');

    fid = fopen(file,'a');
    if fid < 0
        warning('BLLPCS:LogOpenFailed','Cannot open FE log file: %s',file);
        return;
    end
    cleaner = onCleanup(@()fclose(fid));

    if needHeader
        fprintf(fid,'Problem\tRunID\tUpperFE\tTotalLowerFE\n');
    end
    fprintf(fid,'%s\t%s\t%d\t%d\n',problemName,runID,upperFE,totalLowerFE);
end
