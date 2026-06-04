function [archiveX,archiveY] = TruncateArchive(archiveX,archiveY,maxSize)
% Keep a bounded archive with recent samples and a small elite memory.

    if size(archiveX,1) <= maxSize
        return;
    end

    eliteSize = min(round(0.2*maxSize),size(archiveX,1));
    recentSize = maxSize - eliteSize;

    [~,rank] = sort(archiveY,'ascend');
    elite = rank(1:eliteSize);
    recent = (max(1,size(archiveX,1)-recentSize+1):size(archiveX,1))';
    keep = unique([elite(:);recent(:)],'stable');

    if length(keep) > maxSize
        keep = keep(1:maxSize);
    end

    archiveX = archiveX(keep,:);
    archiveY = archiveY(keep,:);
end
