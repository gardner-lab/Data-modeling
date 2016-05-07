% StandardSSGLM.m
% Dec 2015, Yarden Cohen (yardencsmail@gmail.com)
% Runs SSGLM without pattern features on a single unit
%% Declare global variables
global HistMats GammaTimes Delta Nspiking Nbinned GammaLength FeaturesLength Features Ntrials TrialLength BinLength Nbins;
%% Declare local variables
dt=1; %dt=0.001*Delta;
K=Ntrials;
Lambda=zeros(K,TrialLength);
ELambda=Lambda;
Gamma=zeros(GammaLength,1);
Theta0=zeros(Nbins,1);
Sigma=zeros(Nbins);
Theta_k_k_1=zeros(K,Nbins);
Theta_k_k=zeros(K,Nbins);
W_k_k_1=zeros(K,Nbins,Nbins);
W_k_k=zeros(K,Nbins,Nbins);
Theta_k_K=zeros(K,Nbins);
W_k_K=zeros(K,Nbins,Nbins);
A_k=zeros(K,Nbins,Nbins);
W_k_u_K=zeros(K,K,Nbins,Nbins);
Theta_k_K2=zeros(K,Nbins,Nbins);
ExpTheta_K=zeros(K,Nbins);

Impulse_Functions=zeros(TrialLength,Nbins);
for l=1:TrialLength
   r=ceil(l/TrialLength*Nbins);
   Impulse_Functions(l,r)=1;
end

%% EM algorithm constants, thresholds etc'
MAX_GLOBAL_ITERATIONS=20;
EM_CONV_LIMIT=1e-3;
Current_EM_iter=1;
MAX_NEWTON_RAPHSON_ITERATIONS=20;
NEWTON_RAPHSON_CONV_LIMIT=0.01;
EM_stop_criterion=0;
%% Calculate initial values of parameters
wnd=20; %floor(Ntrials/2); %number of trials for regression static GLM.
X=zeros(TrialLength*Ntrials,Nbins+GammaLength);
for k=1:K
    X((k-1)*TrialLength+1:k*TrialLength,1:Nbins)=Impulse_Functions;
    X((k-1)*TrialLength+1:k*TrialLength,(Nbins+1):end)=HistMats{k}(1:1000,:); %fix
end
betas=zeros(Ntrials-wnd+1,Nbins);
for k=1:(K-wnd+1)
    k2=k+wnd-1;
    Y=Nspiking(k:k2,:);
    Y=reshape(Y',wnd*TrialLength,1);
    XX=X((k-1)*TrialLength+1:k2*TrialLength,:); %1:Nbins
    beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
    betas(k,:)=beta(1:Nbins)';
end
Y=Nspiking(:,:);
XX=X(:,:);
Y=reshape(Y',K*TrialLength,1);
beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
Gamma_initial=beta(Nbins+1:end); %beta*0.01; %zeros(size(Gamma));
Gamma_initial(isnan(Gamma_initial))=-5;
% Smooth extreme values
tmpbeta=conv2(betas,ones(3)/9,'same');
locs=find((betas>0) | (betas<-10));
betas(locs)=tmpbeta(locs);
% Initial conditions for Sigma and Theta0, Gamma is set to 0
Theta0_initial=beta(1:Nbins);
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
Gamma=Gamma_initial;
Sigma=Sigma_initial;
itcnt=1;
while (EM_stop_criterion==0)
%% E
% Forward filter
    Theta_k_k_1=zeros(K,Nbins);
    Theta_k_k=zeros(K,Nbins);
    W_k_k_1=zeros(K,Nbins,Nbins);
    W_k_k=zeros(K,Nbins,Nbins);
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
        Lambda(k,:)=exp(Impulse_Functions*Theta_k_k_1(k,:)')'.*exp(HistMat*Gamma)';
        tempW=Impulse_Functions'*Lambda(k,:)';
        W_k_k(k,:,:)=inv(inv(squeeze(W_k_k_1(k,:,:)))+diag(tempW));
        tempW=Impulse_Functions'*(Nspiking(k,:)'-Lambda(k,:)');
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
% Exp Theta
    for k=1:K
        SumWkss=sum(diag(squeeze(W_k_K(k,:,:))));
        for r=1:Nbins
            ExpTheta_K(k,r)=exp(Theta_k_K(k,r))*(1+0.5*W_k_K(k,r,r));%+0.5*(SumWkss-W_k_K(k,r,r)); 
        end
    end
    

%% M
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
            dQdGammaj(j)=sum(sum(-ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j)*dt+Nspiking(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j))); %(:,GammaLength+1:end)
            for s=1:GammaLength
                d2QdGammajdGammas(j,s)=-sum(sum(ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j).*Nspiking(:,GammaLength-s+1:end-s)))*dt;
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


%% history 
% % % % % StandardSSGLM.m Dec2015
% % % % % Jul 2014, Yarden Cohen (yardencsmail@gmail.com)
% % % % % Runs SSGLM without pattern features on a single unit
% % % % %% Declare global variables
% % % % global HistMats GammaTimes Delta Nspiking Nbinned GammaLength FeaturesLength Features Ntrials TrialLength BinLength Nbins;
% % % % %% Declare local variables
% % % % dt=0.001*Delta;
% % % % dt=1; 
% % % % K=Ntrials;
% % % % Lambda=zeros(K,TrialLength);
% % % % ELambda=Lambda;
% % % % Gamma=zeros(GammaLength,1);
% % % % Theta0=zeros(Nbins,1);
% % % % Sigma=zeros(Nbins);
% % % % Theta_k_k_1=zeros(K,Nbins);
% % % % Theta_k_k=zeros(K,Nbins);
% % % % W_k_k_1=zeros(K,Nbins,Nbins);
% % % % W_k_k=zeros(K,Nbins,Nbins);
% % % % Theta_k_K=zeros(K,Nbins);
% % % % W_k_K=zeros(K,Nbins,Nbins);
% % % % A_k=zeros(K,Nbins,Nbins);
% % % % W_k_u_K=zeros(K,K,Nbins,Nbins);
% % % % Theta_k_K2=zeros(K,Nbins,Nbins);
% % % % ExpTheta_K=zeros(K,Nbins);
% % % % 
% % % % Impulse_Functions=zeros(TrialLength,Nbins);
% % % % for l=1:TrialLength
% % % %    r=ceil(l/TrialLength*Nbins);
% % % %    Impulse_Functions(l,r)=1;
% % % % end
% % % % 
% % % % %% EM algorithm constants, thresholds etc'
% % % % MAX_GLOBAL_ITERATIONS=20;
% % % % EM_CONV_LIMIT=1e-3;
% % % % Current_EM_iter=1;
% % % % MAX_NEWTON_RAPHSON_ITERATIONS=20;
% % % % NEWTON_RAPHSON_CONV_LIMIT=0.01;
% % % % EM_stop_criterion=0;
% % % % %% Calculate initial values of parameters
% % % % wnd=20; %floor(Ntrials/2); %number of trials for regression static GLM.
% % % % X=zeros(TrialLength*Ntrials,Nbins+GammaLength);
% % % % for k=1:K
% % % %     X((k-1)*TrialLength+1:k*TrialLength,1:Nbins)=Impulse_Functions;
% % % %     X((k-1)*TrialLength+1:k*TrialLength,(Nbins+1):end)=HistMats{k}(1:1000,:); %fix
% % % % end
% % % % betas=zeros(Ntrials-wnd+1,Nbins);
% % % % for k=1:(K-wnd+1)
% % % %     k2=k+wnd-1;
% % % %     Y=Nspiking(k:k2,:);
% % % %     Y=reshape(Y',wnd*TrialLength,1);
% % % %     XX=X((k-1)*TrialLength+1:k2*TrialLength,:); %1:Nbins
% % % %     beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
% % % %     betas(k,:)=beta(1:Nbins)';
% % % % end
% % % % 
% % % % Y=Nspiking(:,:);
% % % % XX=X(:,:);
% % % % Y=reshape(Y',K*TrialLength,1);
% % % % beta=glmfit(XX,Y,'poisson','log','off',[],[],'off');
% % % % Gamma_initial=beta(Nbins+1:end); %beta*0.01; %zeros(size(Gamma));
% % % % Gamma_initial(isnan(Gamma_initial))=-5;
% % % % 
% % % % % Smooth extreme values
% % % % tmpbeta=conv2(betas,ones(3)/9,'same');
% % % % locs=find((betas>0) | (betas<-10));
% % % % betas(locs)=tmpbeta(locs);
% % % % 
% % % % 
% % % % % Initial conditions for Sigma and Theta0, Gamma is set to 0
% % % % Theta0_initial=beta(1:Nbins);
% % % % 
% % % % NTerms=4;A=1; B=ones(1,NTerms)./NTerms;
% % % % betas(isnan(betas))=0;
% % % % if(size(betas,2)>3*NTerms)
% % % %     fcoeffs = filter(B,A,betas); %filtfilt
% % % % else
% % % %     fcoeffs = betas;
% % % % end
% % % % 
% % % % varEst=nanvar(diff(fcoeffs',[],2),[],2);
% % % % 
% % % % Sigma_initial=diag(varEst);
% % % % 
% % % % % tmp_diff=diff(betas');
% % % % % Sigma_initial=zeros(Nbins);
% % % % % for d=1:Nbins
% % % % %     Sigma_initial(d,d)=var(tmp_diff(:,d));
% % % % % end
% % % % 
% % % % 
% % % % %%
% % % % % Begin EM
% % % % Theta0=Theta0_initial;
% % % % Gamma=Gamma_initial;
% % % % Sigma=Sigma_initial;
% % % % % Theta_k_k(1:size(betas,2),:)=betas';
% % % % % Theta_k_k(size(betas,2)+1:end,:)=ones(K-size(betas,2),1)*betas(:,end)';
% % % % % %Theta_k_k=-20*ones(size(Theta_k_k));
% % % % itcnt=1;
% % % % while (EM_stop_criterion==0)
% % % % %% E
% % % % % Forward filter
% % % %     Theta_k_k_1=zeros(K,Nbins);
% % % %     Theta_k_k=zeros(K,Nbins);
% % % %     W_k_k_1=zeros(K,Nbins,Nbins);
% % % %     W_k_k=zeros(K,Nbins,Nbins);
% % % %     k=1;
% % % %     Theta_k_k_1(1,:)=Theta0';
% % % %     W_k_k_1(1,:,:)=Sigma;
% % % %     %%%
% % % %     for k=1:K
% % % %         if (k==1)
% % % %             Theta_k_k_1(k,:)=Theta0';
% % % %             W_k_k_1(k,:,:)=Sigma;
% % % %         else
% % % %             Theta_k_k_1(k,:)=Theta_k_k(k-1,:);
% % % %             W_k_k_1(k,:,:)=squeeze(W_k_k(k-1,:,:))+Sigma;
% % % %         end
% % % %         HistMat=HistMats{k}(1:TrialLength,:);
% % % %         Lambda(k,:)=exp(Impulse_Functions*Theta_k_k_1(k,:)')'.*exp(HistMat*Gamma)';
% % % %         tempW=Impulse_Functions'*Lambda(k,:)';
% % % %         W_k_k(k,:,:)=inv(inv(squeeze(W_k_k_1(k,:,:)))+diag(tempW));
% % % %         tempW=Impulse_Functions'*(Nspiking(k,:)'-Lambda(k,:)');
% % % %         Theta_k_k(k,:)=Theta_k_k_1(k,:)+(squeeze(W_k_k(k,:,:))*tempW)';
% % % %     %%%
% % % %     
% % % % %          Theta_k_k_1(k,:)=Theta_k_k(k-1,:);
% % % % %          W_k_k_1(k,:,:)=squeeze(W_k_k(k-1,:,:))+Sigma;
% % % % %          HistoryFilter=sameconv(Nspiking(k,:)',Gamma); %check
% % % % %          HistoryFilter=HkAll{k}(1:1000,:)*Gamma;
% % % % %          Lambda(k,:)=exp(Theta_k_k_1(k,:)*Impulse_Functions).*exp(HistoryFilter'); %(GammaLength+1:end)
% % % % %          %%%
% % % % %          tempW=sum(reshape(Lambda(k,:),ceil(TrialLength/Nbins),Nbins));
% % % % %          W_k_k(k,:,:)=inv(inv(squeeze(W_k_k_1(k,:,:)))+dt*diag(tempW));
% % % % %          Theta_k_k(k,:)=Theta_k_k_1(k,:)+(squeeze(W_k_k(k,:,:))*((Nspiking(k,GammaLength+1:end)-Lambda(k,:)*dt)*Impulse_Functions')')';
% % % %     end
% % % % %     for k=1:K
% % % % %         tempW=sum(reshape(Lambda(k,:),ceil(TrialLength/Nbins),Nbins));
% % % % %         W_k_k(k,:,:)=inv(inv(squeeze(W_k_k_1(k,:,:)))+dt*diag(tempW));
% % % % %         Theta_k_k(k,:)=Theta_k_k_1(k,:)+(squeeze(W_k_k(k,:,:))*((Nspiking(k,GammaLength+1:end)-Lambda(k,:)*dt)*Impulse_Functions')')';
% % % % %     end
% % % % % Kalman smoothing
% % % %     Theta_k_K(K,:)=Theta_k_k(K,:);
% % % %     W_k_K(K,:,:)=W_k_k(K,:,:);
% % % %     for k=(K-1):-1:1
% % % %         A_k(k,:,:)=squeeze(W_k_k(k,:,:))*inv(squeeze(W_k_k_1(k+1,:,:)));
% % % %         Theta_k_K(k,:)=Theta_k_k(k,:)+(squeeze(A_k(k,:,:))*(Theta_k_K(k+1,:)-Theta_k_k_1(k+1,:))')';
% % % %         W_k_K(k,:,:)=squeeze(W_k_k(k,:,:))+squeeze(A_k(k,:,:))*(squeeze(W_k_K(k+1,:,:))-squeeze(W_k_k_1(k+1,:,:)))*transpose(squeeze(A_k(k,:,:)));
% % % %     end
% % % % % Theta_K^2
% % % %     for k=1:K
% % % %         Theta_k_K2(k,:,:)=squeeze(W_k_K(k,:,:))+Theta_k_K(k,:)'*Theta_k_K(k,:);
% % % %     end
% % % % % State Space Covariance
% % % %     for k=1:K
% % % %         W_k_u_K(k,k,:,:)=W_k_K(k,:,:);
% % % %     end
% % % %     for u=K:-1:2
% % % %         for k=(u-1):-1:1
% % % %             W_k_u_K(k,u,:,:)=squeeze(A_k(k,:,:))*squeeze(W_k_u_K(k+1,u,:,:));
% % % %             W_k_u_K(u,k,:,:)=squeeze(W_k_u_K(k,u,:,:))'; 
% % % %         end
% % % %     end
% % % % % Exp Theta
% % % %     for k=1:K
% % % %         for r=1:Nbins
% % % %             ExpTheta_K(k,r)=exp(Theta_k_K(k,r))*(1+0.5*W_k_K(k,r,r)); 
% % % %         end
% % % %     end
% % % %     
% % % % 
% % % % %% M
% % % % % new Sigma
% % % %     SigmaNew=squeeze(Theta_k_K2(1,:,:))-Theta0*Theta_k_K(1,:)-Theta_k_K(1,:)'*Theta0'+Theta0*Theta0';
% % % %     for k=2:K
% % % %         SigmaNew=SigmaNew+squeeze(Theta_k_K2(k,:,:))+squeeze(Theta_k_K2(k-1,:,:)) ...
% % % %             -squeeze(W_k_u_K(k,k-1,:,:))-Theta_k_K(k,:)'*Theta_k_K(k-1,:) ...
% % % %             -squeeze(W_k_u_K(k-1,k,:,:))-Theta_k_K(k-1,:)'*Theta_k_K(k,:);
% % % %     end
% % % %     SigmaNew=SigmaNew/K;
% % % %     [v d]=eig(SigmaNew);
% % % %     Sigma=diag(diag(v*d*v'));
% % % % % new Gamma
% % % %     dQdGammaj=zeros(size(Gamma));
% % % %     d2QdGammajdGammas=zeros(GammaLength);
% % % %     Current_NR_Iter=1;
% % % %     GammaOld=Gamma;
% % % %     while ((Current_NR_Iter<=MAX_NEWTON_RAPHSON_ITERATIONS))
% % % %         for k=1:K %flipud
% % % %             HistoryFilter=filter((GammaOld),1,Nspiking(k,:)'); %conv(Nspiking(k,:)',GammaOld); %conv(Nspiking(k,:)',GammaOld); %same
% % % %             ELambda(k,:)=(ExpTheta_K(k,:)*Impulse_Functions').*exp(HistoryFilter(1:end)'); %GammaLength+1,,,GammaLength
% % % %         end
% % % %         for j=1:GammaLength
% % % %             dQdGammaj(j)=sum(sum(-ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j)*dt+Nspiking(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j))); %(:,GammaLength+1:end)
% % % %             for s=1:GammaLength
% % % %                 d2QdGammajdGammas(j,s)=-sum(sum(ELambda(:,GammaLength+1:end).*Nspiking(:,GammaLength-j+1:end-j).*Nspiking(:,GammaLength-s+1:end-s))); %*dt
% % % %             end
% % % %         end
% % % %         GammaNew=GammaOld-inv(d2QdGammajdGammas)*dQdGammaj;
% % % %         scaling=max(abs(GammaNew));
% % % %         if (scaling>4)
% % % %             GammaNew=GammaNew/scaling*4;
% % % %         end
% % % %         Current_NR_Iter=Current_NR_Iter+1;
% % % %         if (max(abs(GammaNew-GammaOld))<NEWTON_RAPHSON_CONV_LIMIT)
% % % %             Current_NR_Iter=MAX_NEWTON_RAPHSON_ITERATIONS+1;
% % % %         end
% % % %         GammaOld=GammaNew;
% % % %     end
% % % %     Gamma=GammaNew;
% % % % % new Theta0
% % % %     Theta0New=Theta_k_K(1,:)';
% % % %     Theta0=Theta0New;
% % % %     imagesc(Theta_k_K); title(num2str(itcnt)); drawnow;
% % % %     itcnt=itcnt+1;
% % % %     
% % % % end



