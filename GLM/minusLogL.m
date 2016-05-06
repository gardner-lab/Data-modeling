function [f g]=minusLogL(x)
global SumLogLambda LogLambda dt;
f=-SumLogLambda*x+sum(exp(LogLambda*x))*dt;
g=(-SumLogLambda+sum(LogLambda.*repmat(exp(LogLambda*x),1,length(x))*dt));
    