library(lhs)
library(mlegp)
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

fit_g  <- mlegp(
    xx,y, 
    constantMean = 0, nugget = NULL, nugget.known = 0, min.nugget = 0, 
    param.names = NULL, gp.names = NULL, PC.UD = NULL, PC.num = NULL, PC.percent = NULL, 
    simplex.ntries = 5, simplex.maxiter = 500, simplex.reltol = 1e-8,  
    BFGS.maxiter = 500, BFGS.tol = 0.01, BFGS.h = 1e-10, verbose = 0, parallel = FALSE
)

## Testing data
result_rmspe_sd = numeric(100)
for (i in 1:100) {
    x.new<-randomLHS(m,p)
    new <-lower+x.new*(upper-lower)
    truey<-new[,1]*sin(new[,1])
    pred_g <-predict(fit_g,x.new)
    RMSPE_sd_g <- sqrt(mean((c(pred_g)-truey)^2))/sd(truey)
    result_rmspe_sd[i]=RMSPE_sd_g
} 

print(mean(result_rmspe_sd))
write.csv(result_rmspe_sd,'./RMSPE/Toy/mleGP-XSinX-LinearMean.csv',row.names=FALSE)