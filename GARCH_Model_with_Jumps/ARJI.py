# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:55:10 2026

@author: user
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.special import expit, logit, logsumexp
from scipy.stats import poisson, norm
from scipy.optimize import minimize
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/"data"
OUTPUT_DIR=BASE_DIR/"output"
OUTPUT_DIR.mkdir(exist_ok=True)

def log_likeli(params,r_t,j_max):
    mu, log_omega, raw_alpha, raw_beta, theta, log_delta, log_lambda0, raw_rho, raw_gamma=params
    omega=np.exp(log_omega)
    delta=np.exp(log_delta)
    lambda0=np.exp(log_lambda0)
    alpha=expit(raw_alpha)
    beta=(1-alpha)*expit(raw_beta)
    rho=expit(raw_rho)
    gamma=rho*expit(raw_gamma)
    
    lambda_t=lambda0/(1-rho)
    h_t=omega/(1-alpha-beta)
    xi_t=0
    
    ll=0
    n_t_list=[]
    h_t_list=[]
    lambda_t_list=[]
    xi_t_list=[]
    jump_prob_list=[]
    
    for i in range(len(r_t)):
        rt=r_t[i]
        log_density=0
        
        jumps=np.arange(j_max+1)
        mean_j=mu+theta*(jumps-lambda_t)
        var_j=h_t+jumps*delta**2
        log_poisson_prob=poisson.logpmf(jumps, mu=lambda_t)
        log_condi_density=norm.logpdf(
            rt,
            loc=mean_j,
            scale=np.sqrt(var_j)
            )
        log_component=log_poisson_prob+log_condi_density
        log_density=logsumexp(log_component)
        ll+=log_density
        
        #算n_hat
        posterior=np.exp(log_component-log_density)
        #jump_prob
        jump_prob_list.append(1-posterior[0])
        n_hat=np.sum(jumps*posterior)
        #算當期xi_t
        xi_t=n_hat-lambda_t
        
        n_t_list.append(n_hat)
        h_t_list.append(h_t)
        xi_t_list.append(xi_t)
        lambda_t_list.append(lambda_t)
        
        #更新參數
        h_t=omega+alpha*(rt-mu)**2+beta*h_t
        lambda_t=lambda0+rho*lambda_t+gamma*xi_t
    return ll, n_t_list, h_t_list, lambda_t_list, xi_t_list, jump_prob_list

def neg_ll(params,r_t,j_max):
    ll=log_likeli(params,r_t,j_max)[0]
    return -ll

#設定初始解
def init(r_t):
    mu=np.mean(r_t)
    alpha=0.05
    beta=0.90
    omega=np.var(r_t,ddof=1)*(1-alpha-beta)
    theta=0.0
    delta=2*np.sqrt(np.var(r_t,ddof=1))
    rho=0.80
    lambda0=0.045*(1-rho)     #假設的數字為長期平均每天約有多少 jump
    gamma=0.10

    init=np.array([
        mu,
        np.log(omega),
        logit(alpha),
        logit(beta/(1-alpha)),
        theta,
        np.log(delta),
        np.log(lambda0),
        logit(rho),
        logit(gamma/rho)
        ])
    return init

def mini(r_t,j_max):
    init_params=init(r_t)
    result=minimize(
        neg_ll,
        init_params,
        args=(r_t, j_max),
        method="L-BFGS-B",
        options={
            "maxiter": 200,
            "maxfun": 3000,
            "ftol": 1e-8
            }
        )
    
    #參數轉換回來
    mu, log_omega, raw_alpha, raw_beta, theta, log_delta, log_lambda0, raw_rho, raw_gamma=result.x
    omega=np.exp(log_omega)
    delta=np.exp(log_delta)
    lambda0=np.exp(log_lambda0)
    alpha=expit(raw_alpha)
    beta=(1-alpha)*expit(raw_beta)
    rho=expit(raw_rho)
    gamma=rho*expit(raw_gamma)
    
    ll, n_t_list, h_t_list, lambda_t_list, xi_t_list, jump_prob_list=log_likeli(result.x,r_t,j_max)
    
    params_result={
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        "delta": delta,
        "lambda0": lambda0,
        "rho": rho,
        "gamma": gamma,
        "Log-likelihood": -result.fun,
        "success": result.success,
        "message": result.message,
        }
    
    filtered_result=pd.DataFrame({
        "return": r_t,
        "h_t": h_t_list,
        "conditional_sd": np.sqrt(h_t_list),
        "lambda_t": lambda_t_list,
        "n_hat": n_t_list,
        "xi_t": xi_t_list,
        "jump_prob_list": jump_prob_list
    })
    return params_result, filtered_result

##############################執行區域##############################
ticker=yf.Ticker("^TWII")
price=ticker.history(
    start="2001-01-01",
    end="2021-01-01",
    interval="1d",
    auto_adjust=True)     #配股、拆股與股利調整

r_t=np.log(price["Close"]/price["Close"].shift(1))
r_t=r_t.dropna()
date=r_t.index
r_t=r_t.to_numpy(dtype=float)

params_result, filtered_result=mini(r_t,5)
filtered_result.index=date

filtered_result.to_csv(
    OUTPUT_DIR/"filtered_result.csv",
    index=True,
    encoding="big5"
    )