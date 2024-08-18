#%%
import numpy as np
import math
class Poissonmethod:
    @staticmethod
    def poissonexpression(X,beta):
        return np.exp(np.dot(X,beta))
    @staticmethod
    def pmf(k,mu):
        return (mu**k * math.exp(-mu)) / math.factorial(k)
    @staticmethod
    def loglikelihood(X,beta,y):
        mu=Poissonmethod.poissonexpression(X,beta)
        log_likelihood = np.sum(y * np.log(mu) - mu- np.log(np.array([math.factorial(i) for i in y])))
        return log_likelihood
    @staticmethod
    def MLEwithL1(X,beta,y,L1):
        loglikelihood=Poissonmethod.loglikelihood(X,beta,y)
        return -loglikelihood+L1*np.sum(np.abs(beta))
    def gradient(X,beta,y,L1):
        mu=Poissonmethod.poissonexpression(X,beta)
        gradident=np.dot(X.T,(y-mu))
        l1=L1*np.sign(beta)
        return -gradident+l1

    @staticmethod
    def d_beta_d_lambda(X, beta, y, L1):
        sign_beta = np.sign(beta)
        d_beta_d_lambda = np.zeros_like(beta)

        for j in range(len(beta)):
            numerator = sign_beta[j]
            denominator = np.sum(X[:, j] ** 2 * np.exp(X @ beta))
            d_beta_d_lambda[j] = numerator / denominator

        return d_beta_d_lambda
#%%
    
#%%
X = np.array([[1, 1], [1, 2], [1, 3]])  # 设计矩阵，带有常数项
y = np.array([1, 2, 3])  # 观测值
l1=10
# 初始参数
beta= np.zeros(X.shape[1])
beta=np.full(X.shape[1],1)
k = 3
lambda_ = 2.5
expression = Poissonmethod.poissonexpression(X,beta)
probability =Poissonmethod.pmf(k,lambda_)
loglikelihood = Poissonmethod.loglikelihood(X,beta,y)
MLE=Poissonmethod.MLEwithL1(X,beta,y,l1)
gradient=Poissonmethod.gradient(X,beta,y,l1)
print(f'P(X = {k}) = {probability}')
print(expression)
print(loglikelihood)
print(MLE)
print(beta)
print(gradient)
#%%
