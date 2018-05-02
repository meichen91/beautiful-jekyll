---
layout: post
title: Machine learning - Week 2
---

This is the second full-time week spent on things besides my PhD and time has been distributed between algorithms and machine learning. Cramming the algorithms (from computer science) has surprised me about how little I know 'behind the scene'. More importantly, I've become motivated to absorb powerful design principles and grasp the useful data structures. 

Motivated by the success last week, I took similar approaches and found the open course by [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/index.htm){:target="_blank"} as a good starting material. The instructors are great! Highly recommend Prof. Demaine's lectures on dynamical programming. Some of the assignments are quite challenging but most of them funny too. This one on [image resize](https://www.youtube.com/watch?v=vIFCV2spKtg&list=LLCWoDukC6wT5KpS9Yq9YTTA&t=0s&index=1){:target="_blank"}  amazed me!

## Results summary
### Theory notes
- [Bishop Notes](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Notes/BishopNotes.pdf){:target="_blank"} updated
	- EM for factor analysis
	- Probabilistic graphical models fundamentals
	- Hidden Markov Model
- [Reinforcement learning](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Notes/ReinforcementLearning.pdf){:target="_blank"}
	- Basics concepts on Markov decision process based on the lecture notes of [CS229](http://cs229.stanford.edu/syllabus.html){:target="_blank"}
- [Algorithms](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Notes/Algorithms.pdf){:target="_blank"}
	- Notes from  [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/index.htm){:target="_blank"}
### Notebooks
- [K means for image compression](https://github.com/meichen91/MachineLearning-Snippets/blob/master/CS229_PS/PS3_Q5_KMeans_Compression.ipynb){:target="_blank"}
	- Applied accelerated vectorised implementation
	- EM interpretation
- [Reinforcement Learning of inverted pendulum](https://github.com/meichen91/MachineLearning-Snippets/blob/master/CS229_PS/PS4_Q6_ReinforcementLearning.ipynb){:target="_blank"}
	- Basic implementation of Markov Decision Process
- [Hidden Markov Model](https://github.com/meichen91/MachineLearning-Snippets/blob/master/ML_ToyExamples/HMM_FeverModel.ipynb){:target="_blank"}
	- Toy Example from [Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm){:target="_blank"}
	- Implementation of the forward-backward algorithm for computing posterior probabilities and Viterbi algorithm for finding the optimal path
	- Learning for the multinomial example needs further work.

### Some thoughts
Been wondering: how important is the understanding of algorithm to practice machine learning (ML)? On the one hand, one would most likely work on a team and there will be software developer to handle the problem. On the other hand, I feel strongly about the power of knowing the fundamentals well. The reasons are: (1) Efficient implementation is desirable and programming logic and intuitions are useful! (2) Some ML algorithms might benefit from certain data structures/algorithm design. E.g. the reinforcement learning and dynamic programming. (3) Really helps understanding complexity of a program, which comes handy when scaling things up.

Probabilistic graphica models are beautiful! Last time I mentioned the Bayesian perspective and the visualisation of such models enlighten me once more. Highly recommend Bishop (2006) §8.1 showing the graphical model on the same example as §1.2.5.
### Things to follow up on
-  There are a bigger picture behind PCA! Such as probabilistic PCA, kernel PCA. Really tempted to dig down as I love the basic form!
- **Variational methods** is essential for intractable problems such as some extended HMM. 
