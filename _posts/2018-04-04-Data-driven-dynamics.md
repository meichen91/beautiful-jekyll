---
layout: post
title: Data-driven Dynamics - What and Why?
---

I will tell my personal story how I came to my passion about data-driven dynamics and the training I have been acquiring. 

## What is Data-Driven Dynamics? 
Data-driven dynamic, in my perspective, is about distilling 'physical law' purely from data. Once a friend joked about its role for liberating physicist from ever to derive equations again. To begin with, this tool has proved its power in solving the well-known physical laws, even in chaotic systems, like the double-pendulum and lorenz attractor. It also scales to high-dimensional data when researchers harnessed the sparsity of natural laws. 

The more exciting aspect is to discover new laws, which is particularly helpful for neuroscience and other biological systems, climate science and economics, because we lack the fundamental understanding of the system. For example, the algorithm may be able to derive the 'law' of brain by analysing the EEG data readily available. 

Although it is a cliché, I have to admit that I am motivated by how much it can contribute in the Big Data era. The field dynamical system and control is well-established; there are flourishing research in machine learning that process data at massive scale and in novel ways. Data-driven dynamics could leverage the knowledge from the conventional field, combine with new machine learning tools, to make sense of the vast data.

## How I became interested?

It was a difficult time of my PhD when I had loads of data but was struggling to interpret them. I came to know the great tool principle component analysis (PCA) (called proper orthogonal decomposition in fluid dynamics). Out of the seemingly random data, beautiful  patterns emerge from applying PCA. It triggered a positive feedback loop of learning and applying. I then naturally become interested in dynamic mode decomposition (DMD), which combines PCA with Fourier transform, another extremely powerful tool.

I could not fully understand the algorithm so I started building my vocabulary via self-paced learning. After taking the [courses](#OnlineCourses) on linear dynamical system and control, mechanical  engineering analysis, and chaos theory, I became deeply intrigued by the elegant interprations from linear algebra and its profound implication for engineering.

After this, I tried using DMD and its variants on data and gained a new, brilliant perspective on turbulence. 

## Getting serious?
I am exploring this research area, especially the major analysis tools, on my own. I try to balance the searching,  learning, implementation and exploitation (e.g., writing my tech blog). The approach that works particularly well so far, is
 1. Search curiously
 2. Narrow down an interesting topic, read and take notes
 3. Pick a good paper, take good notes. Implement the algorithm on toy problems first and play with it.
 4. If possible, implement relevant techniques, compare and learn.

Most often, I identified holes in my knowledge and the following learning would be really rewarding, such as picking up the basics of nonlinear time-series analysis.

I had an interesting dataset to begin with and used this approach to learn (1) spectral-POD, (2) the numerous variants of DMD, and (3) koopman mode decomposition. Then I explored sparse sensing out of interests.

## Conclusion
I do not recommend this way of doing research, because one needs proper training during their PhD. However, I have benefitted immensely from this learning experience. The lessons I have learnt are: 
1. To take ownership of my research
2. Great works take time and effort, (and are risky,) so they take cycles.
3. To go through the cycles calls for  plans with deliverables (e.g., report, mini-project) to keep me focused and track the progress.
4. Build positive feedback loop.

I got taught by amazing lectures over the world and met one of my idols because of my passion. I hope to share with the public  about this exciting field through my blog and I do appreciate any request or contribution.
## Apendices
### Online courses:
<a name="OnlineCourses"></a>
**[Introduction to Linear Dynamical Systems](http://ee263.stanford.edu/)** EE263 Stanford, [Video lectures](https://www.youtube.com/watch?v=bf1264iFr-w&list=PL06960BA52D0DB32B) by Stephen Boyd
	Great lecturer! The course work is really useful. Being able to see the meaning of the linear algebra operations  helps my understanding. 

**[Control Bootcamp](https://www.youtube.com/watch?v=Pi7l8mMjYVE&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m)** by Steve Brunton, University of Washington
Amazing lecturer with brand-new way of lecturing! Really clear teaching and plenty of implementations to play with.

**[Nonlinear Dynamics and Chaos](https://www.youtube.com/watch?v=ycJEoqmQvwg&list=PLbN57C5Zdl6j_qJA-pARJnKsmROzPnO9V)** by Steven Strogatz, Cornell University
The lectures follow his book (beautifully written!) and I have mainly relied on his [book](https://www.amazon.co.uk/Nonlinear-Dynamics-Chaos-Studies-Nonlinearity/dp/0813349109/ref=pd_lpo_sbs_14_t_0?_encoding=UTF8&psc=1&refRID=ZYCM406DPR37DACQCRFR). His teaching leads you slowly into the difficult concepts and explained things really well. I wish I had began studying chaos, especially bifurcation, from this book!

**[Mechanical Engineering Analysis](https://www.youtube.com/watch?v=QM0ATZRlbKQ&list=PLMrJAkhIeNNR2W2sPWsYxfrxcASrUt_9j)**: [ME564](http://faculty.washington.edu/sbrunton/me564/) and [
ME565](http://faculty.washington.edu/sbrunton/me565/) by Steve Brunton, University of Washington
Really rigorous and clear teaching on how to solve ODE and PDE. More relevant to engineers. 

**[Nonlinear Dynamics: Geometry of Chaos](http://chaosbook.org/course1/about.html)** by Predrag Cvitanović from.
Georgia Institute of Technology.
The courses accompany the [chaos book](http://chaosbook.org/) and is mathematically very challenging.
### Research groups
In my opinion, the [Kutz group](http://faculty.washington.edu/kutz/) and [Brunton Lab](https://www.eigensteve.com/) and their co-workers at University of Washington are leading the frontier in this direction.
The other researchers  I identified are [Hayden Schaeffer](http://math.cmu.edu/~hschaeff/) from CMU and [Maziar Raissi](http://www.dam.brown.edu/people/mraissi/) from Brown University. 

### Future plan
The tools that I am most familiar with are DMD and compressed sensing. Therefore, I will exploit these advantages, get a few more miniprojects done before applying for jobs. 
The other interesting topics via the 'curious' search include 
 - Using various machine learning techniques to model chaos theory
 - Predictive model based on chaos theory (e.g., work by [Chris Danfort](http://www.uvm.edu/~cdanfort/main/home.html))
 - The synergy between dynamical system and work from Yutian Chen and Max Welling on herded Gibbs sampling is also intriguing. 
