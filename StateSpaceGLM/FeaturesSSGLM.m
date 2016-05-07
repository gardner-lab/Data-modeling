% FeaturesSSGLM.m
% Dec 2015, Yarden Cohen (yardencsmail@gmail.com)
% Runs SSGLM with pattern features on a single unit
%% Declare global variables
global HistMats GammaTimes Delta Nspiking Nbinned GammaLength NFeatures Features Ntrials TrialLength BinLength Nbins;
%% Declare local variables
dt=1; %dt=0.001*Delta;
Delta=1;
K=Ntrials;
Lambda=zeros(K,TrialLength);
ELambda=Lambda;
Gamma=zeros(GammaLength,1);
Theta0=zeros(Nbins*NFeatures,1);
Sigma=zeros(Nbins*NFeatures);
Theta_k_k_1=zeros(K,Nbins*NFeatures);
Theta_k_k=zeros(K,Nbins*NFeatures);
W_k_k_1=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
W_k_k=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
Theta_k_K=zeros(K,Nbins*NFeatures);
W_k_K=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
A_k=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
W_k_u_K=zeros(K,K,Nbins*NFeatures,Nbins*NFeatures);
Theta_k_K2=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
ExpTheta_K=zeros(K,Nbins*NFeatures);
FeaturesMat=zeros(K,Nbins*NFeatures);
for k=1:K
    for r=1:Nbins
        FeaturesMat(k,(r-1)*NFeatures+1:r*NFeatures,(r-1)*NFeatures+1:r*NFeatures)=Features(k,:)'*Features(k,:);
    end
end
Impulse_Functions=zeros(TrialLength,Nbins*NFeatures);
for l=1:TrialLength
    r=ceil(l/TrialLength*Nbins);
    Impulse_Functions(l,(r-1)*NFeatures+1:r*NFeatures)=1;
end
%Feature_Functions=Impulse_Functions'.*repmat(Features,1,TrialLength/NFeatures);


%% EM algorithm constants, thresholds etc'
MAX_GLOBAL_ITERATIONS=20;
EM_CONV_LIMIT=1e-3;
Current_EM_iter=1;
MAX_NEWTON_RAPHSON_ITERATIONS=50;
NEWTON_RAPHSON_CONV_LIMIT=0.01;
EM_stop_criterion=0;
%% Calculate initial values of parameters
wnd=20; %floor(Ntrials/2); %number of trials for regression static GLM.
X=zeros(TrialLength*Ntrials,Nbins*NFeatures+GammaLength);
for k=1:K
    X((k-1)*TrialLength+1:k*TrialLength,1:Nbins*NFeatures)=Impulse_Functions.*repmat(Features(k,:),TrialLength,Nbins);
    X((k-1)*TrialLength+1:k*TrialLength,(Nbins*NFeatures+1):end)=HistMats{k}(:,:); %fix
end
betas=zeros(Ntrials-wnd+1,Nbins*NFeatures);
for k=1:(K-wnd+1)
    k2=k+wnd-1;
    Y=Nspiking(k:k2,:);
    Y=reshape(Y',wnd*TrialLength,1);
    XX=X((k-1)*TrialLength+1:k2*TrialLength,:); %1:Nbins
    beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
    betas(k,:)=beta(1:Nbins*NFeatures)';
end
Y=Nspiking(:,:);
XX=X(:,:);
Y=reshape(Y',K*TrialLength,1);
beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
Gamma_initial=beta(Nbins*NFeatures+1:end); %beta*0.01; %zeros(size(Gamma));
Gamma_initial(isnan(Gamma_initial))=-5;
% Smooth extreme values
tmpbeta=conv2(betas,ones(3)/9,'same');
locs=find((betas>0) | (betas<-10));
betas(locs)=tmpbeta(locs);
% Initial conditions for Sigma and Theta0, Gamma is set to 0
Theta0_initial=beta(1:Nbins*NFeatures);
NTerms=4;A=1; B=ones(1,NTerms)./NTerms;
betas(isnan(betas))=0;
if(size(betas,2)>3*NTerms)
    fcoeffs = filter(B,A,betas); %filtfilt
else
    fcoeffs = betas;
end
varEst=nanvar(diff(fcoeffs',[],2),[],2);
Sigma_initial=diag(varEst);
% Begin EM
Theta0=Theta0_initial;
Theta0(abs(Theta0)>0.1)=0.1;
Theta0(isnan(Theta0))=0.001;
Gamma=Gamma_initial;
Gamma(abs(Gamma)>4)=-4;
Sigma=Sigma_initial;
itcnt=1;
Sigma(Sigma>0.1)=0.1;
Sigma(isnan(Sigma))=0.0001;
%%
while (EM_stop_criterion==0)
% E
% Forward filter
    Theta_k_k_1=zeros(K,Nbins*NFeatures);
    Theta_k_k=zeros(K,Nbins*NFeatures);
    W_k_k_1=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
    W_k_k=zeros(K,Nbins*NFeatures,Nbins*NFeatures);
    k=1;
    Theta_k_k_1(1,:)=Theta0';
    W_k_k_1(1,:,:)=Sigma;
    for k=1:K
        if (k==1)
            Theta_k_k_1(k,:)=Theta0';
            W_k_k_1(k,:,:)=Sigma;
        else
            Theta_k_k_1(k,:)=Theta_k_k(k-1,:);
            W_k_k_1(k,:,:)=squeeze(W_k_k(k-1,:,:))+Sigma;
        end
        HistMat=HistMats{k}(1:TrialLength,:);
        Lambda(k,:)=exp((Impulse_Functions.*repmat(Features(k,:),TrialLength,Nbins))*Theta_k_k_1(k,:)')'.*exp(HistMat*Gamma)';
        tempW=Impulse_Functions'*Lambda(k,:)'*Delta;
        W_k_k(k,:,:)=inv(inv(squeeze(W_k_k_1(k,:,:)))+squeeze(FeaturesMat(k,:,:))*diag(tempW));
        tempW=Impulse_Functions'*(Nspiking(k,:)'-Lambda(k,:)'*Delta).*repmat(Features(k,:)',Nbins,1);
        Theta_k_k(k,:)=Theta_k_k_1(k,:)+(squeeze(W_k_k(k,:,:))*tempW)';
    end
% Kalman smoothing
    Theta_k_K(K,:)=Theta_k_k(K,:);
    W_k_K(K,:,:)=W_k_k(K,:,:);
    for k=(K-1):-1:1
        A_k(k,:,:)=squeeze(W_k_k(k,:,:))*inv(squeeze(W_k_k_1(k+1,:,:)));
        Theta_k_K(k,:)=Theta_k_k(k,:)+(squeeze(A_k(k,:,:))*(Theta_k_K(k+1,:)-Theta_k_k_1(k+1,:))')';
        W_k_K(k,:,:)=squeeze(W_k_k(k,:,:))+squeeze(A_k(k,:,:))*(squeeze(W_k_K(k+1,:,:))-squeeze(W_k_k_1(k+1,:,:)))*transpose(squeeze(A_k(k,:,:)));
    end
% Theta_K^2
    for k=1:K
        Theta_k_K2(k,:,:)=squeeze(W_k_K(k,:,:))+Theta_k_K(k,:)'*Theta_k_K(k,:);
    end
% State Space Covariance
    for k=1:K
        W_k_u_K(k,k,:,:)=W_k_K(k,:,:);
    end
    for u=K:-1:2
        for k=(u-1):-1:1
            W_k_u_K(k,u,:,:)=squeeze(A_k(k,:,:))*squeeze(W_k_u_K(k+1,u,:,:));
            W_k_u_K(u,k,:,:)=squeeze(W_k_u_K(k,u,:,:))'; 
        end
    end
% Exp Theta - need to check
    for k=1:K
        for r=1:Nbins
            ExpTheta_K(k,(r-1)*NFeatures+1:r*NFeatures)=exp(Theta_k_K(k,(r-1)*NFeatures+1:r*NFeatures).*Features(k,:))*(1+0.5*Features(k,:)*squeeze(W_k_K(k,(r-1)*NFeatures+1:r*NFeatures,(r-1)*NFeatures+1:r*NFeatures))*Features(k,:)'); 
        end
    end
    

% M
% new Sigma
    SigmaNew=squeeze(Theta_k_K2(1,:,:))-Theta0*Theta_k_K(1,:)-Theta_k_K(1,:)'*Theta0'+Theta0*Theta0';
    for k=2:K
        SigmaNew=SigmaNew+squeeze(Theta_k_K2(k,:,:))+squeeze(Theta_k_K2(k-1,:,:)) ...
            -squeeze(W_k_u_K(k,k-1,:,:))-Theta_k_K(k,:)'*Theta_k_K(k-1,:) ...
            -squeeze(W_k_u_K(k-1,k,:,:))-Theta_k_K(k-1,:)'*Theta_k_K(k,:);
    end
    SigmaNew=SigmaNew/K;
    [v d]=eig(SigmaNew);
    Sigma=diag(diag(v*d*v'));
% new Gamma
    dQdGammaj=zeros(size(Gamma));
    d2QdGammajdGammas=zeros(GammaLength);
    Current_NR_Iter=1;
    GammaOld=Gamma;
    while ((Current_NR_Iter<=MAX_NEWTON_RAPHSON_ITERATIONS))
        for k=1:K %flipud
            HistMat_loc=HistMats{k}(1:TrialLength,:);
            HistoryFilter=HistMat_loc*GammaOld;%filter((GammaOld),1,Nspiking(k,:)'); %conv(Nspiking(k,:)',GammaOld); %conv(Nspiking(k,:)',GammaOld); %same
            ELambda(k,:)=(ExpTheta_K(k,:)*Impulse_Functions').*exp(HistoryFilter(1:end)'); %GammaLength+1,,,GammaLength
        end
        for j=1:GammaLength
            dQdGammaj(j)=sum(sum(-Delta*ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j)*dt+Nspiking(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j))); %(:,GammaLength+1:end)
            for s=1:GammaLength
                d2QdGammajdGammas(j,s)=-sum(sum(Delta*ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j).*Nspiking(:,GammaLength-s+1:end-s)))*dt;
            end
        end
        GammaNew=GammaOld-inv(d2QdGammajdGammas)*dQdGammaj;
        scaling=max(abs(GammaNew));
        if (scaling>4)
            GammaNew=GammaNew/scaling*4;
        end
        Current_NR_Iter=Current_NR_Iter+1;
        if (max(abs(GammaNew-GammaOld))<NEWTON_RAPHSON_CONV_LIMIT)
            Current_NR_Iter=MAX_NEWTON_RAPHSON_ITERATIONS+1;
        end
        GammaOld=GammaNew;
    end
    Gamma=GammaNew;
% new Theta0
    Theta0New=Theta_k_K(1,:)';
    Theta0=Theta0New;
    imagesc(Theta_k_K); title(num2str(itcnt)); drawnow;
    itcnt=itcnt+1;
    
end