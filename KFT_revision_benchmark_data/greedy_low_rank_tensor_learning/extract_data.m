dat = open("USHCN.mat");
locations = dat.locations;
csvwrite('side_info.csv',locations)
data = dat.data_series;

X = zeros(331500,3);
Y = zeros(331500,1);
counter = 1;
for i=1:1:17
    tmp_view = data{i};
    for loc = 1:1:125
       for t=1:1:156
          X(counter,:) = [i,loc,t];
          Y(counter,1)=tmp_view(loc,t);
          counter=counter+1;
       end 
    end
end
csvwrite('X.csv',X)
csvwrite('Y.csv',Y)


