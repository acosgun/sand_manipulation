clear;clc;
data = load("/home/acosgun/git/sand_manipulation/data/dataU.mat")
dataY = data.dataY;
dataU = data.dataU;
deltaUX = data.deltaUX;
deltaUY = data.deltaUY;
deltaYX = data.deltaYX;
deltaYY = data.deltaYY;

%csvwrite('/home/acosgun/git/sand_manipulation/data/dataY.dat',dataY)
%csvwrite('/home/acosgun/git/sand_manipulation/data/dataU.dat',dataU)

net = fitnet(100);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net,tr] = train(net, dataU', dataY');
dataY_nn = net(dataU');
errors = gsubtract(dataY, dataY_nn');