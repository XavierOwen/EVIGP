library(lhs)
library(GPfit)
library(abind)

set.seed(0)

rm(list=ls())
lower<-0.
upper<-10.

p<-1
m<-100
n<-20

## Training data
xx<-maximinLHS(n,p)

x<-scale(xx,F,1/(upper-lower))
x<-scale(x,-lower,F)    ## x is in the range of the phyiscal model

y<-x[,1]*sin(x[,1])

y <- y+ rnorm(n, mean = 0, sd=0.5)

fit_g <- GP_fit(xx,y,nug_thres=25,corr=list(type='exponential',power=2))

## Testing data
result_rmspe_sd = numeric(100)

for (i in 1:100) {
    x.new<-randomLHS(m,p)
    new <-lower+x.new*(upper-lower)
    truey<-new[,1]*sin(new[,1])
    pred_g <-predict(fit_g,xnew=x.new)
    RMSPE_sd_g <- sqrt(mean((pred_g$Y_hat-truey)^2))/sd(truey) 
    result_rmspe_sd[i]=RMSPE_sd_g
} 

print(mean(result_rmspe_sd))
write.csv(result_rmspe_sd,'./RMSPE/Toy/GPfit-XSinX-ConstMean.csv',row.names=FALSE)