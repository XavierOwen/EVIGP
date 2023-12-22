library(lhs)
library(mlegp)
library(abind)

set.seed(0)

rm(list=ls())
lower<-c(0.05,100,63070,990,63.1,700,1120,9855)
upper<-c(0.15,50000,115600,1110,116,820,1680,12045)

p<-8
m<-1000
n<-160

## Training data
xx<-maximinLHS(n,p)

x<-scale(xx,F,1/(upper-lower))
x<-scale(x,-lower,F)

y<-2*pi*x[,3]*(x[,4]-x[,6])/(log(x[,2]/x[,1])*(1+2*x[,7]*x[,3]/(log(x[,2]/x[,1])*x[,1]^2*x[,8])+x[,3]/x[,5]))   ## No noise involved.

y <- y+ rnorm(n, mean = 0, sd=0.02)

fit_g  <- mlegp(
    xx,y, 
    constantMean = 1, nugget = NULL, nugget.known = 0, min.nugget = 0, 
    param.names = NULL, gp.names = NULL, PC.UD = NULL, PC.num = NULL, PC.percent = NULL, 
    simplex.ntries = 5, simplex.maxiter = 500, simplex.reltol = 1e-8,  
    BFGS.maxiter = 500, BFGS.tol = 0.01, BFGS.h = 1e-10, verbose = 0, parallel = FALSE
)

## Testing data
result_rmspe_sd = numeric(100)
for (i in 1:100) {
    x.new<-randomLHS(m,p)
    new <-lower+x.new*(upper-lower)
    truey<-2*pi*new[,3]*(new[,4]-new[,6])/( log(new[,2]/new[,1])* (1+2*new[,7]*new[,3]/(log(new[,2]/new[,1])*new[,1]^2*new[,8])+new[,3]/new[,5]) )
    pred_g <-predict(fit_g,x.new)
    RMSPE_sd_g <- sqrt(mean((c(pred_g)-truey)^2))/sd(truey)  
    result_rmspe_sd[i]=RMSPE_sd_g
} 

print(mean(result_rmspe_sd))
write.csv(result_rmspe_sd,'./RMSPE/Borehole/mleGP-BH-ConstMean.csv',row.names=FALSE)
