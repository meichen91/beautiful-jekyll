---
layout: post
title: Survival analysis I - core concepts
---

For the first project of my new job, I got a very exciting task, which belongs to the realm of survival analysis. As I was new to this field, I spent quite some time learning and really enjoyed the process. One of the first obstacles I found was trying to get my head around hazard function, survival function, many different residuals etc. This motivates me to summarise what I learnt in this blog. However, I am not an expert in this topic and I look forward to learning more. If there is any correction or suggestion, I would appreciate any feedback!

## What is survival analysis?
Survival analysis, in essence, studies time to event. To give it some context in analyzing patients' survival time, we are interested in questions like what proportion of patients survived after a given time? What factors affected patitents' survival? 

Essentially, it is a regression task. Instead of simply predicting how long each patient will survive, we are more ambitous. We would like to derive the survival probability at each time while only observing the final outcome. 

It is applied in lots more fields besides the very fruitful medical domain, including the failure of a mechanical part, customer churn, time until re-employment.

### Definitions
There are some reoccuring terms in this topic, 
- Event: what we are interested in predicting, described as binary.
- Time: how long has passed since the beginning of observation.
- Censoring: if a patient survived during the observation time, we refer to them as censored. We can imagine that if death is unavoidable, their time are 'censored' by the experiment. This scenario is often referred to as right-censored and left-censoring means that an event is known to have occurred before a certain time.

### Training data
Given that we have defined the terms, it would be a good time to imagine what the data looks like. 

|patient| age | treatment | ...| time (years)  | status|
|--|--|--|--|--|--|--|
| A | 67 | Yes | ...| 4 | 1 |
| B | 60 | No | ... | 2 | 0 |
| C  | 60 | No | ...| 5 | 1 |

For each patient, we have some information that may or may not affect the survival and the time and outcome of the event.

## Modelling framework
Our goal (getting the survival probability, potentially taylored for each subject) is quite ambitious but the survival often follows some natural law, governed by certain underlying mechanism. Therefore, we expect certain theretical framework to be useful, which often involves survival & hazard functions.
To begin with the simpler one, survival function is the probability of surviving a certain time. From a frequentist perspective, it is the proportion of patients surviving after a certain time. In mathematical terms, if we denote the survival time as $T$

$$S(t) = \text{Prob}\{T > t\}$$

Hazard function $\lambda(t)$ is also called instantaneous event (death, failure) rate. In contrast to survival function as a cumulative effect, it captures the instantaneous effect. If we think of hazard function as the velocity, survival function is like the distance travelled. 

There is slightly complication because to study the rate of failure at a certain time $t$, we implicitly assumed that the subject has survived up until this point. We express this in terms of conditional probability and express the failure rate using classic calculus formulation

$$ \lambda(t) = \lim_{u \rightarrow 0} \frac{\text{Prob}\{t < T \leq t + u \vert T > t\}}{u}$$

If we replace $u$ with $dt$, it may appear more familar with folks with some physics background. 

### Relationship between  survival & hazard functions
If you think of the speed/distance analogy, knowing one gives you full information of the other. Both are very useful, $S(t)$ gives the probability that speaks to our intuition and applies to further cost/benefit analysis while $\lambda(t)$ is likely governed by some underlying mechanism. Their relationship is not a trivial derivation but a very beautiful one.

First we apply the law of conditional probability $P(A\vert B) = P(A)/P(A,B)$

$$\lambda(t) =  \lim_{u \rightarrow 0} \frac{\text{Prob}\{t < T \leq t + u\}/\text{Prob}\{T > t\}}{u}$$

We can replace the probability using the definition of $S(t) = \text{Prob}\{T > t\}$

$$\lambda(t) =  \lim_{u \rightarrow 0} \frac{[S(t) - S(t + u)]/u}{S(t)}$$

Recognizing that the top of the expression can be replaced with the derivative,

$$\lambda(t) = -\frac{\partial S(t)/\partial t}{S(t)} $$

which can be rewritten as

$$ \lambda(t) = - \frac{\partial \ln S(t)}{\partial t} $$

To derive $S(t)$ using $\lambda(t)$, we have

$$\int_0^t \lambda(\nu) d\nu = \Lambda(t) = - \ln S(t) $$

where $\Lambda(t)$ is known as the cumulative hazard function.

### Why is it useful?
From a practical persepctive, the above relationship is useful because we can compute the other when the package we chose only provides one of the prediction. In the theoretical framework, this logarithmic formula is commonly used; combined with further exponential relationships when we investigate the effect of other factors, extra care is needed.

## Coming-up next
In this blog, I laid the framework for further discussion as survival analysis is not widely known. More importantly, I found the understanding of the relationship between the survival & hazard functions very helpful. In the next one, I shall introduce the major models and assessment metrics for model fit. The final blog consists of an example (programmed in R), so we can see the theory in action.

## References
The best material I found is the *Regression Modelling Strategy* by Frank E. Harrell. The textbook has detailed derivation and great explanation and comes with R code to play with. I have also read several blogs about this topic. 
