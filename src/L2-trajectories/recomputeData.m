function allData = recomputeData( allData )

% allData=alldata;
for icam=1:4
    index = icam*4+1;
    avg_width = mean(allData(allData(:, 7)~=-1, 7));
    avg_height = mean(allData(allData(:, 8)~=-1, 8));
    f
    
end