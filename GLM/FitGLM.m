global SumLogLambda LogLambda dt;
% Implements GLM for spiking history, other units' effect and stimuli
% effect.
% Prepare a file called GLMdata.mat that contains two sparse matrices:
%
%  Raster: a sparse matrix with one column for each neuron and one row for
%  each time bin. When neuron 'a' gives a spike at time bin 'b' set
%  Raster(b,a)=1;
%
% Stim: a sparse matrix with one column for each stimulus type and one row
% for each time bin. If the value of stimulus 'a' is 's' at time bin 'b'
% set Stim(b,a)=s;


dt=0.001;                   % the time bins of the data (Seconds)
UnitNum=3;                  % the # of the unit to optimize for

% The neuron's influence on itself
OwnParamsNum=10;            % Number of filters
AbsoluteRefractory=0.01;   % Duration of minimal refractory period (Seconds)
OwnLastPeak=0.5;            % Location of last peak in the filters basis (Seconds)
OwnFirstPeak=0.04;          % Location of first peak in the filters basis (Seconds)
OwnBumpsSpacing=0.5;        % Sets the distribution of bumps (small --> non linear spacing and widths)

% Other neurons' influence
OtherParamsNum=10;          % Number of filters
OtherLastPeak=0.5;          % Location of last peak in the filters basis (Seconds)
OtherFirstPeak=0.0;         % Location of first peak in the filters basis (Seconds)
OtherBumpsSpacing=0.5;      % Sets the distribution of bumps (small --> non linear spacing and widths)

% External stimuli influence
StimParamsNum=10;           % Number of filters
StimLastPeak=1.2;             % Location of last peak in the filters basis (Seconds)
StimFirstPeak=0.06;          % Location of first peak in the filters basis (Seconds)
StimBumpsSpacing=0.5;       % Sets the distribution of bumps (small --> non linear spacing and widths)

% optimization settings
% change the number after 'MaxIter' to the desired number of iterations 
options=optimset('MaxIter',50,'Display','iter','GradObj','on');


% Create filters
ihbasprs.ncols = OwnParamsNum;  
ihbasprs.hpeaks = [OwnFirstPeak OwnLastPeak];  
ihbasprs.b = OwnBumpsSpacing;  
ihbasprs.absref = AbsoluteRefractory;  %% (optional)
[ih1t,ih1bas,ih1basis] = makeBasis_PostSpike(ihbasprs,dt);

ihbasprs.ncols = OtherParamsNum;  
ihbasprs.hpeaks = [OtherFirstPeak OtherLastPeak];  
ihbasprs.b = OtherBumpsSpacing;  
ihbasprs.absref = .00;  %% (optional)
[iht,ihbas,ihbasis] = makeBasis_PostSpike(ihbasprs,dt);

ihbasprs.ncols = StimParamsNum;  
ihbasprs.hpeaks = [StimFirstPeak StimLastPeak];  
ihbasprs.b = StimBumpsSpacing;  
ihbasprs.absref = .00;  %% (optional)
[kht,kbas,kbasis] = makeBasis_PostSpike(ihbasprs,dt);

% Load spikes and stimuli 
BinsToIgnore=[];
load GLMdata;

% Convolve data with filters and prepare matrices for optimization
nn=size(Raster,2);
ns=size(Stim,2);
ExpLng=size(Raster,1);

H=zeros(ExpLng,(nn-1)*OtherParamsNum+OwnParamsNum);
K=zeros(ExpLng,StimParamsNum*ns);
curr_unit=1;
for un=[1:(UnitNum-1) (UnitNum+1):nn]
    for par=1:OtherParamsNum
        H(:,(curr_unit-1)*OtherParamsNum+par)=sameconv(full(Raster(:,un)),ihbas(:,par));
    end
    curr_unit=curr_unit+1;
end

for par=1:OwnParamsNum
    H(:,(nn-1)*OtherParamsNum+par)=sameconv(full(Raster(:,UnitNum)),ih1bas(:,par));
end

for st=1:ns
    for par=1:StimParamsNum
        K(:,(st-1)*StimParamsNum+par)=sameconv(full(Stim(:,st)),kbas(:,par));
    end
end

locs=find(full(Raster(:,UnitNum))==1);
locs=setdiff(locs,BinsToIgnore);
SumLogLambda=[sum([H(locs,:) K(locs,:)]) length(locs)];
locs=setdiff(1:ExpLng,BinsToIgnore);
LogLambda=[H(locs,:) K(locs,:) ones(length(locs),1)];

clear H K;

% Perform optimization
[bet L]=fminunc(@(x)minusLogL(x),zeros((nn-1)*OtherParamsNum+OwnParamsNum+ns*StimParamsNum+1,1),options);

% Outputs
OtherNeuronsFilters=ihbas*reshape(bet(1:(nn-1)*OtherParamsNum),OtherParamsNum,nn-1);
OwnFilter=ih1bas*bet((nn-1)*OtherParamsNum+1:(nn-1)*OtherParamsNum+OwnParamsNum);
StimuliFilters=kbas*reshape(bet((nn-1)*OtherParamsNum+OwnParamsNum+1:end-1),StimParamsNum,ns);
BaseLineRate=exp(bet(end));
Lambda=exp(LogLambda*bet);

figure; plot(ih1t,OwnFilter); title(['Self influence filter. Baseline rate is ' num2str(BaseLineRate) ' sp/s']);
if (nn>1)
    figure; plot(iht,OtherNeuronsFilters); title('Other Neurons'' influence');
end
if (ns>0)
    figure; plot(kht,StimuliFilters); title('Stimuli influence');
end

figure;
plot(sameconv(full(Raster(:,UnitNum)),ones(1000,1)/1000),'r');
hold on
plot(sameconv(exp(LogLambda*bet)*dt,ones(1000,1)/1000),'b');
legend({'smoothed spikes' 'fit'})