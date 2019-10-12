library("ipred")
library("survival")
library("survivalROC")
library("glmnet")
library('kernlab')
library('caret')
setwd('D:/pan-cancer/brca')
par(mfrow = c(3, 4))
########################LIHC#########################
kx2=read.csv('brca_cox2.csv',row.names= 1)
ind=5
results_fin<-matrix(nrow = ind, ncol=6)
j=6
set.seed(j)
k=10
folds <- createFolds(as.data.frame(t(kx2)),k)
results<-matrix(nrow = k, ncol=4)
eva<-matrix(nrow = nrow(kx2), ncol=k)
count=0
for(i in 1:k){
  testset <- kx2[folds[[i]],] 
  trainset <- kx2[-folds[[i]],] 
  x=trainset[,-c(1:2)]
  x=as.matrix(x)
  tc=trainset$time
  tc[tc==0]=0.001
  y=Surv(tc,trainset$status)
  cv.fit<-cv.glmnet(x,y,family="cox",maxit=10000,alpha=0,nfold=5)
  fit<-glmnet(x,y,family="cox",alpha=0)
  tt=predict(fit,x,s=cv.fit$lambda.min)
  x2=testset[,-c(1:2)]
  x2=as.matrix(x2)
  tc2=testset$time
  tc2[tc2==0]=0.001
  y2=Surv(tc2,testset$status)
  tt2=predict(fit,x2,s=cv.fit$lambda.min)
  ci_k2=survConcordance(formula = y2~ tt2)
  
  
  results[i,1]<-ci_k2$concordance
  eva[(count+1):(count+nrow(tt2)),1]<-tc2
  eva[(count+1):(count+nrow(tt2)),2]<-testset$status
  eva[(count+1):(count+nrow(tt2)),3]=tt2
  count=count+nrow(tt2)
  
}

tc3=eva[,1]/365
st3=eva[,2]
rk3=eva[,3]
rk4=eva[,4]
y3=Surv(tc3,st3)
ci_k3=survConcordance(formula = y3~ rk3)
ci_k3$concordance
mm=median(rk3)
for (p in 1:length(rk3)){
  if (rk3[p]>mm){
    rk3[p]=1 
  } else  
  {rk3[p]=-1}
}
q2=survdiff(y3~rk3)
p.val <- 1 - pchisq(q2$chisq, length(q2$n) - 1)
p.val
sd11=survfit(y3~rk3)




################################################
kt2=kx2[,-(1:2)]
cl <- kmeans(kt2, 2)
riskk=cl$cluster
yy=Surv((kx2$time)/365,kx2$status)
sp=survdiff(yy~riskk)
p.valk2 <- 1 - pchisq(sp$chisq, length(sp$n) - 1)
p.valk2
riskk[riskk==2]=0
#riskk_bl=riskk
sd12=survfit(yy~riskk_bl)


plot(sd11,conf.int=FALSE,mark.time=T,col=c("green","red"),lty=1,main="BRCA",
     cex.main=1.5,xlab="Year",ylab = "Survival Probablity",cex.lab=1.4,cex.axis=1.2,lwd=2)
lines(sd12,mark.time=T,col=c("grey","grey"),lty=2,lwd=2)
legend('topright',c('DCAP-low risk','DCAP-high risk'),lty=c(1,1),col=c(3,2),cex=1.2)