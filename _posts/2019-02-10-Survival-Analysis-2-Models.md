---
layout: post
title: Survival analysis II - key models and performance metrics
---

Last blog introduced the core concepts and terminology in survival analysis and the two central quantities for modelling the survival process, the survival and hazard functions. This time, I will continue the journey in the theoretical land, covering some key models and performance metrics.

## Key models
### Non-parametric models
The basic scenario is to ignore the individual characteristics and assuming that the survival probability is only a function of time. The first one we will look at is the Kaplan–Meier (KM) estimator. This is the most widely used model and can be derived from the maximum likelihood model. The survival probability give the data is computed as
$$S(t + 1) = S(t) \text{Prob}(\text{Survived past } t + 1 | \text{Survived past } t )$$

Here I will give a simple numerical example as illustration. Suppose that we are given the following data:

|    Time      |    4    |    7    |    8    |    11    |    13    |    15    |    16    |    17    |    19    |    21    |
|--------------|---------|---------|---------|----------|----------|----------|----------|----------|----------|----------|
|    Status    |    1    |    1    |    0    |    1     |    0     |    1     |    1     |    0     |    1     |    0     |

We will summarise the data as the following format

|    Time    |    n.risk    |    n.event    |    n.censored    |    Survival function      |
|------------|--------------|---------------|------------------|---------------------------|
|    4       |    10        |    1          |    0             |    9/10 = 0.9             |
|    7       |    9         |    1          |    0             |    0.9 * 8/9 = 0.8        |
|    11      |    7         |    1          |    1             |    0.8 * 6/7 = 0.686      |
|    15      |    5         |    1          |    1             |    0.686 * 4/5 = 0.549    |
|    16      |    4         |    1          |    0             |    0.549 * 3/4 = 0.411    |
|    19      |    2         |    1          |    1             |    0.411 * 1/2 = 0.206    |


In the summary table, we have the number of events happened at each time point, the number of censored 'events' occuring since the last time point. Subtracting these two numbers from the number of people survived by this time point, i.e., `n.risk`, we have the `n.risk` for the next time.  $\text{Prob}(\text{Survived past } t + 1 \vert \text{Survived past } t )$ can be expressed as $\frac{ \text{n.risk} - \text{n.event} }{\text{n.risk}}$ using the frequentist formula. Looking more closely at this calculation, we can see that the censored data contribute to the survival probability for the computation before their 'survival time' but they are ignored for the computation afterwards.

 The other option, the Nelson–Aalen estimator models the hazard function instead of the survival probability. The hazard function, i.e., the instantaneous failure rate is calculated as
 $\lambda (t_i ) = \frac{ \text{n.event} }{\text{n.risk}}$. 

According to the derivation in the last blog, $\lambda(t) = - \frac{\partial \ln S(t)}{\partial t}$. The survival function is thus
 $S(t) = \exp(-\int_0^t \lambda(\nu) d\nu) = \exp(-\sum_0^t(\lambda (t_i )))$. Using the same data, the resulting estimation is given below: 

 |    Time    |    n.risk    |    n.event    |    n.censored    |   Hazard function|  Survival function       |
|------------|--------------|---------------|------------------|------------|---------------|
|    4       |    10        |    1          |    0             | 1/10|   exp(-1/10) = 0.905            |
|    7       |    9         |    1          |    0             |  1/9|   exp(-(1/10 + 1/9)) = 0.810       |
|    11      |    7         |    1          |    1             |  1/7|   0.810  * exp(-1/7) =0.702    |
|    15      |    5         |    1          |    1             | 1/5|   0.702  * exp(-1/5) = 0.575   |
|    16      |    4         |    1          |    0             |  1/4|   0.575 * exp(-1/4) = 0.448    |
|    19      |    2         |    1          |    1             |  1/2|   0.448 * exp(-1/2) = 0.271     |

It should be noted that the difference between two estimations are negligible when the sample size is large.[^1]

### Cox proportional hazard model
The non-parametric model gives an estimation of the time dependency. Ultimately, we would like to come up with a prediction of survival probability for each subject, based on the individual characteristics. We use  a predictor vector to represent individual characteristics, noted as $X$. Cox proportional hazard model is a semi-parametric model to quantify the effect of other variables, e.g., age, sex, the presence of a treatment. The assumption is that the factors act multiplicatively on the failure rate and the impact does not change with time. This multiplication is often in exponential form, written as

$$\lambda(t|X) = \lambda(t)\exp(X\beta) = \lambda(t)\exp(x_1\beta_1 + x_2\beta_2 + ... ) = \lambda(t)\exp(x_1\beta_1)\exp(x_2\beta_2) ...  $$

This decoupling between time and other factors is the most important assumption, aka, proportional hazard (PH) assumption. It means that the hazard ratio of two individuals, $\frac{\lambda_1 (t)}{\lambda_2 (t)}$, is constant in time.  Plotting $\ln \lambda_1 (t)$ and $\ln \lambda_2 (t)$ for two populations (e.g., male & female) helps assess the proportionality assumption because $\ln \lambda_1 (t)$ and  $\ln \lambda_2 (t)$ should be parallel given the proportionality assumption.

The cumulative hazard function scales the same way

$$\Lambda(t \vert X) = \Lambda(t)\exp(X\beta) $$

while the survival function scales as

$$S(t\vert X) = \exp[-\Lambda(t)\exp(X\beta)] = \exp[-\Lambda(t)]^{\exp(X\beta)}$$

To assess the PH assumption, we can plot the equivalent of $\ln \lambda (t)$ as $\ln(-\ln(S(t\vert X))$

### Parametric proportional hazard models

Cox proportional hazard model is called semi-parametric because the time-dependent term, $\lambda(t)$, is estimated from data, usually using KM estimator. Many parametric PH models are used to exploit underlying mechanism. For example, the exponential regression model assumes that $\lambda(t)$ is a constant, i.e., the failure rate does not change with time. It is known as exponential regression model because the survival probability decreases exponentially. 

Another popular model is the Weibull regression model, characterised by hazard function of the following form:

$$\lambda(t \vert X) = \alpha \gamma t^{\gamma - 1} \exp(X\beta)$$

### Accelerated failure time models
In contrast to the proportional hazard models, the accelerated failure time (AFT) models uses time as the response variable, instead of modelling the hazard function. AFT models assume that the logarithmic of the failure time ($\ln(T)$) is a linear function of the predictors.

$$\ln(T) = \beta_0 + \beta_1 X_{1} + ... + \beta_p X_{p}$$

With an increase of 1 unit in $X_j$, the predicted $T$ is multiplied by $\exp(\beta_j)$, giving the amount of acceleration/deceleration of the failure time. For proportional hazard models, when $X_j$ increases 1 unit, the hazard function is multiplied by $\exp(\beta_j)$.

## Model evaluations
### Comparing survival functions
The log-rank test is the most widely used method of comparing two or more survival curves. The null hypothesis is that there is no difference in survival between the two groups.  Under this null hypothesis, the log-rank test statistics is asymptotically distributed as $\chi^2$. 

### ROC and AUC
To gain a more intuitive understanding of the performance of the fitted models, we can use area-under-curve (AUC). Although survival analysis is a regression task, we can also interpret the outcome as a classification task at each time point. At a certain time point, survival models give a survival probability of each individual and we know whether they have survived at that time. 

For example, at a given threshold, we predict 800 survivals and 200 failures. Out of the 800 survival, 720 are in the survived class (true positive); out of the 200 failures, 30 are in the survived class (false negative). Thus the true positive rate is $720/(720 + 30) = 0.96$ and the false positive rate is $80/(80 + 170) = 0.32$. Repeating this with different thresholds, a ROC can be computed. To compare two fitted survival models, we can use the time-averaged AUC.

### Residuals
For regression tasks, the residual is often used to quantify the error and diagnose the model fitting. Residual, defined as the the expected minus the predicted value, is not obvious in survival analysis. The difficulties in definition include censored data, lack of knowledge of survival probability etc. In the literature, several different residuals are proposed for different purposes. The following are the most commonly used:

|Residual| Purpose |
|--|--|
| Martingale residual | model fitting  |
|Schoenfeld residual| PH assumption|
|Deviance residual| outliers|
|Score residual| influential points|


## Coming next
In this blog, we introduced the major survival models. A detailed description of the non-parametric models is given to reinforce the concepts from the first blog. To model the effect of predictors, proportional hazard models and AFT models are often used. They are fitted using maximum likelihood estimation, which I have not touched on. Several model evaluation metrics are briefly introduced, with more details in the next blog. I will also show the models in action, using R packages, and will definitely include some figures! 

## References
Besides *Regression Modelling Strategy*, I have also drawn insights from _Applied survival analysis_ on various topics. For the different residuals, I have found [these lecture notes](https://www.ics.uci.edu/~dgillen/STAT255/Handouts/lecture10.pdf) useful.


[^1]: Hosmer, David W., Stanley Lemeshow, and Susanne May. _Applied survival analysis_. Wiley Blackwell, 2011.
