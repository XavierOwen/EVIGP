library(lhs)
library(laGP)
library(abind)

set.seed(0)

rm(list=ls())
lower<-0.
upper<-10.

p<-1
m<-100
n<-11

## Training data
xx<-maximinLHS(n,p)

x<-scale(xx,F,1/(upper-lower))
x<-scale(x,-lower,F)    ## x is in the range of the phyiscal model

y<-x[,1]*sin(x[,1])

y <- y+ rnorm(n, mean = 0, sd=0.5)

## Testing data
result_rmspe_sd = numeric(100)
for (i in 1:100) {
    x.new<-randomLHS(m,p)
    new<-matrix(0,ncol=p,nrow=m)
    for (j in 1:m)
        new[j,]<-lower+x.new[j,]*(upper-lower)
    truey<-new[,1]*sin(new[,1])
    fit_g = aGP(xx, y, x.new, start = 6, end = 10, d = NULL, g = 1/10000,method = c("alc", "alcray", "mspe", "nn", "fish"), Xi.ret = TRUE, verb=0);
    pred_g <-fit_g$mean
    RMSPE_sd_g <- sqrt(mean((c(pred_g)-truey)^2))/sd(truey)
    result_rmspe_sd[i]=RMSPE_sd_g
} 

print(mean(result_rmspe_sd))
write.csv(result_rmspe_sd,'./RMSPE/Toy/laGP-XSinX-ConstMean.csv',row.names=FALSE)