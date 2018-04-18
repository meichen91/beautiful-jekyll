---
layout: post
title: Machine learning summary: Week 1
---

In order to prepare for interviews and get myself educated about this exciting field. I have started 'cramming' the fundamentals in the field. Although there are many cool and complicated tools in libraries and I can use them in a plug-and-go fashion, I do not find this approach satisfying. I want to know why things work. 

Personally, I believe that clear understanding of the fundamentals will be powerful in the long run. One clear indicator is to be able to explain things intuitively, as in the **Feynman Technique**. This belief determines the goal and strategy of my learning.

A great thing about machine learning is that there are so many great materials online and there is an active community. Because the aim is to understand the fundamentals and I enjoy learning by doing, I have found the following way for the first round:

 1. Learn the theory.
 2. Come up with a toy example to test things out, poke around, and test the understanding.
 3. If available, try a more realistic and complex data set. 

##  Learning materials
### Prior knowledge
- I took the [machine learning MOOC by Andrew Ng](https://www.coursera.org/learn/machine-learning) 4 years ago ( I wish I had known the importance of fundamentals and intuitions at that time.)
- Linear algebra is really useful and my experience is mainly from studying dynamical system theory.
- Some probabilities and stats.
- Machine learning: I have learnt a little about deep learning (ANN, CNN) and Bayesian networks. Had research experience with compressed sensing. Very familiar with PCA. 

### Week 0

 - Chapters 1 & 2 of *Bishop (2006): Pattern Recognition and Machine Learning*
 The book is very well-written but I do not find it a great material for 'cramming'. 
 - Udacity: [Intro into Data Science](https://classroom.udacity.com/courses/ud170)
The Ucacity course gave me some practice with numpy and pandas and helped me transit from using MatLab.

### Week 1
- Stanford [CS229 Machine Learning](http://cs229.stanford.edu/syllabus.html)
I mainly follow this Stanford Course because it was cleverly designed to introduce students to the field, so it means rigor and structure. (Also there are problem sheets to work through.) The other benefits is to get an motivation.
- Bishop (2006)
Sometimes, using multiple materials can be confusing because of the different notations. I refer to Bishop (2006) to get a second perspective and answer some questions based on the lecture notes. The book is also a source of linking the concepts together
- Googling
There are lots of videos, python notebooks, and tutorials online. The Scikit learn websites have many useful examples that are lightweight and insightful.

## Results summary
### Theory notes:
- Bishop Notes
		-  k-means clustering and Expectation maximisation
		-  k-Nearest Neighbours density estimation
- SVM
		- I find the [SVM lecture in MIT 6.034 Artificial Intelligence course](16.%20Learning:%20Support%20Vector%20Machines%20-%20YouTube) to be a very helpful introduction. The Prof is really good at building from the fundamentals and telling a story!
- Neural Network
		- A personal distillation of the [Deep Learning Tutorial from Stanford](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/).
		- The online book (http://neuralnetworksanddeeplearning.com/) is also helpful.
### Python notebooks
1. [Logistic regression](https://github.com/meichen91/MachineLearning-Snippets/blob/master/CS229_PS/PS2_Q1_LogisticRegression_TrainingStability.ipynb)
- Basic gradient descent implementation
- Insightful example about the importance of 'normalisation' such as that used in SVM
2.  [Naive Bayes](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Week1/NaiveBayes.ipynb)
- Toy example to check understanding
3. [SVM](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Week1/SVM_Exploration.ipynb) 
- Simple example to illustrate high-dimensional space using 2nd-order polynomial.
- Visualise decision boundaries using grid.
4. [k-Nearest neighbours](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Week1/KNN_Vectorisation.ipynb)
- Example from CS229 on the importance of vectorisation
- Time saving is astonishing!
5. [2 layer neural network with logistic regression](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Week1/NeuralNet_2Layer.ipynb)
- Stochastic and vectorised implementation
- Numerical gradient verification
- MNIST test
6. [2 layer neural network with soft-max regression](https://github.com/meichen91/MachineLearning-Snippets/blob/master/Week1/NN_2Layer_Softmax.ipynb)
- MNIST test (the accuracy is not satisfactory). 


### Other interesting things

 - Highly recommend [the Talking Machines](https://www.thetalkingmachines.com/) podcast! I just learnt the EM before stubling upon an old episode on it. The explanation was consice and insightful. Also I learnt that people are using linear dynamical systems (LDS) in machine learning, such as in combination with hidden Markov Model. The example Ryan talked about echoes with my idea of using LDS with robotics. 
 - The [Microsoft Research podcast](https://www.microsoft.com/en-us/research/blog/category/podcast/) is also really interesting. The breath of research and impact of the 'grand goal' is fascinating.
 - Finished the book [Weapons of maths destruction](https://weaponsofmathdestructionbook.com/) recently. The book promotes building fairness, transparency, etc, into the algorithms. It explains the ideas behind the recent headline stories (Facebook and Cambridge Analytica*) and many more.
\*This company has nothing to do with the University of Cambridge!

###  Other thoughts
- Began to understand why Neural networks are so versatile.
- Vectorisation is extremely powerful! Although the vectorised version of the 2-layer NN looks similar to the stochastic one, it took me a whole night (but I find it worthwhile, for the happiness when it works :))
- I used to think that we only need the right cost function and smart implementation of the optimisation. It is important. However, this two weeks of learning revealed a more insightful picture from the Bayesian perspective. It was an epiphany when I read about least square cost function being the maximum likelihood solution with Gaussian noise.

### Some ideas
- Visualisation of NN and the potential application of sparsity in NN.
- Using SVM for the low-dimensional representation of the Dog-Cat classification. See if it improve classification using [SPOCC](http://meichenlu.com/2018-04-01-Compressed-sensing-for-classification/) vectors.