library(lhs)
library(mlegp)
library(abind)

set.seed(0)

rm(list=ls())
lower<-c(50, 25,0.5,1.2,0.25,50)
upper<-c(150,70,3,  2.5,1.2,300)

p<-6
m<-1000
n<-200

## Training data
xx<-maximinLHS(n,p)

x<-scale(xx,F,1/(upper-lower))
x<-scale(x,-lower,F)

Vbl <- 12*x[,2]/(x[,1]+x[,2])
termOne <- (Vbl+0.74)*x[,6]*(x[,5]+9)/(x[,6]*(x[,5]+9)+x[,3])
termTwo <- 11.35*x[,3]/(x[,6]*(x[,5]+9)+x[,3])
termThree <- 0.74*x[,3]*x[,6]*(x[,5]+9)/(x[,6]*(x[,5]+9)+x[,3])/x[,4]

y<- termOne + termTwo + termThree

y <- y+ rnorm(n, mean = 0, sd=0.02)

## insert second order effects
xx <- cbind(xx,xx^2)

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
    truey<-2*pi*new[,3]*(new[,4]-new[,6])/( log(new[,2]/new[,1])* (1+2*new[,7]*new[,3]/(log(new[,2]/new[,1])*new[,1]^2*new[,8])+new[,3]/new[,5]) )
    x.new <- cbind(x.new,x.new^2)
    pred_g <-predict(fit_g,x.new)
    RMSPE_sd_g <- sqrt(mean((c(pred_g)-truey)^2))/sd(truey)  
    result_rmspe_sd[i]=RMSPE_sd_g
} 

print(mean(result_rmspe_sd))
write.csv(result_rmspe_sd,'./RMSPE/OTLcircuit/mleGP-OTL-QuadraticMean.csv',row.names=FALSE)
