Download Link: https://assignmentchef.com/product/solved-ceng499-homework-4-naive-bayes-hidden-markov-models
<br>
In this assignment, you have 2 independent tasks: sentiment analysis using Naive Bayes and evaluation and decoding tasks of Hidden Markov Models. No report is required in this homework. All related files can be downloaded from http://user.ceng.metu.edu.tr/~artun/ceng499/hw4_files.zip.

1           Sentiment Analysis using Naive Bayes1.1          DatasetThe dataset is created using Twitter US Airline Sentiment dataset. You need to use the version we provided. The examples are some tweets that mention some airline firms in US. They are labeled as negative, neutral, and positive and given as plain text files in hw4_data/sentiment.

1.2          Background1.2.1         Naive BayesLet h be our hypothesis which is defined as

h(x) = argmaxP(y|x),

y

where y is the label and x is the feature. It basically selects the label that has the highest probability given x. Using the Bayes’ Rule, we can obtain

h(x) = argmaxP(y|x)

y

Bayes’ Rule

= argmaxP(x|y)P(y)                                       P(x) does not depend on y;

y

however, estimating P(x|y) is still hard since we need to see the exact example in out dataset. In our case, it is highly unlikely that we see the exact same test tweet in our training dataset. To achieve more feasible solution, Naive Bayes assumes conditional independence between the dimensions of the features given label. Although this independence does not hold in our case in reality, it still works well in practice. Assuming the independence, we achieve

d

h(x) = argmaxP(y) Y P(xj|y)

y j=1

where xj is the jth dimension of the feature x and d is the number of dimensions of x. In general, it is not a good idea to multiply many probabilities because our precision may not be enough to represent it and the operation may result with a 0. Instead, we can take its logarithm (which does not change the results since it is a monotonically increasing function) and get

d

h(x) = argmaxlog(P(y) Y P(xj|y))

y j=1 d

= argmaxlog(P(y)) + Xlog(P(xj|y)).

y j=1

1.2.2         Our caseWe are going to assume that the positions of the words do not matter and simply count the occurrences of the words in an example. This kind of modelling is called bag-of-words. To do that, first, we are going to create our vocabulary which will contain every word in the training set. (We could also use every word in the English dictionary but in this homework, we are going to use the training data we already have.) Secondly, for a specific example (a tweet), we will count the occurrences of every word in the vocabulary.

To give an example let’s say our tweet is “the ball is on the floor”. Then, our representation will be something like in Table 1 assuming that our vocabulary is {a, amazing, ball, floor, is, on, the, zoom}.

a0 amazing0 ball1 floor1 is1 on1 the2 zoom0

Table 1: The representation of “the ball is on the floor”.

We have two probabilities to estimate: P(x|y) and P(y). P(y) is easy to estimate. For class c we will count the examples labeled as class c and divide it by the number of all examples. Let πc = P(y = c) be the probability of a tweet being in class c. Then, it can be estimated as

where I is the indicator function which returns 1 when the condition holds and 0 otherwise, y(i) is the label of ith example in the dataset, and n is the number of examples in the training dataset.

In our model, we will think like while writing a tweet, every word is selected by rolling a die with d sides where d is the number of words in our vocabulary. Let θjc be the probability of rolling the word j in our vocabulary given that our class label is c. Then, the probability of generating a tweet with length

m is Multinomial Distribution (https://en.wikipedia.org/wiki/Multinomial_distribution) and can be expressed as

.

Since we are taking argmax, we don’t actually need to calculate the term   since it will be the same for every class. Therefore, finally our hypothesis function will be

d

h(x) = argmaxlog(πˆc) + Xxjlog(θˆjc)

c

j=1

where θˆjc is the estimation of θjc. We will see how to estimate that in the next section.

1.2.3         Additive (Laplace) SmoothingCalculation of θˆjc’s will be very similar to the calculation of πˆc’s. We will simply divide the number of occurrences of the jth word in all examples labeled with class c by the number of total words in the examples labeled with c. More formally, it can be described as

where y(i) is the label of the ith example, x(ji) is the number of occurrences of the jth word in our vocabulary in the ith example. The term  simply calculates the number of words in the ith example.

Let’s say the word “cake” appears in some negative tweets but never appears in a positive tweet in our training set. Then, if we estimate θ’s like we did in above, the estimated probability of the word “cake” appearing in a positive tweet would be 0. Since we multiply the probabilities, the overall probability will be 0. (Since we are taking the logarithm of them, we can get a domain error.) However, we may get a test example where the word “cake” appears in it while being a positive tweet. So we want to somehow smooth the probabilities so that even it does not occur in our training set, the probability will be some small value, but not 0.

In additive smoothing with smoothing parameter 1 (https://en.wikipedia.org/wiki/Additive_ smoothing), we are going to behave like every word appears at least once in the training dataset. If we update our estimation formula above, the number of every word will increase by one so the denominator will increase by the number of words in our vocabulary (d) and the numerator will increase by 1 and we will get

.

1.3          ImplementationYou could do your implementations by directly coding the formulas above, it would be a very inefficient since it contains many unnecessary calculations. This homework does not require you to implement very efficient code; however, your code still needs to finish in a reasonable amount of time. (for example in 10 minutes). If you don’t remove the redundant things, the calculations may far exceed this time limitation.

Notice that although the equations seem very complicated, all it does is counting and dividing the results to the total number of what it counts. To estimate πc, we can count how many times a specific label c occurs in the training data and then divide it by the the length of our the training set. Similarly, to estimate θ:c, we can count how many times every word occurs in the training data labeled with specific class c by only traversing it once and divide the values by the total number of words for class c. Additive smoothing can also be easily adapted to this kind of implementation. If you get rid of the redundancies in your code in this way, your implementation should be able to finish its job under 1 second.

1.4          TaskYou are expected to train a Naive Bayes classifier, test it using the test set and report the accuracy. The function templates are given in the nb.py and some tests are given in the nb_mini_test.py. Passing all these tests does not mean you will get full points.

•    Extract the words in the sentences. Since these tweets are not preprocess, you can clean up your data by removing the special characters. However, some of them may be important to the classification; for example, the emojis. This part of the homework will not be graded however, you still need to convert the sentences into words so that you can use them in your naive bayes algorithm. You can simply divide your sentences from whitespaces if you want to.

•    Implement the functions in nb.py according to their descriptions. During test time, you can skip the words that are not in the vocabulary created using the train set. The scores of the test function should be calculated using (as mentioned above)

d

h(x) = argmaxlog(πˆc) + Xxjlog(θˆjc).

c

j=1

•    Calculate the accuracy of your model using the functions in the nb.py on the test set and print it.

2           Hidden Markov ModelsIn this part, you are going to work on evaluation and decoding tasks of Hidden Markov Models (HMM). To do that, you are going to implement forward and Viterbi algorithms. The template of the functions are given in “hmm.py”. You can check the recitation slides for the detailed explanations of forward and Viterbi algorithms and understand the notation used here.

2.1          DataOnly filling the forward and viterbi functions is enough for this homework. Arguments will be directly fed into those functions. Therefore, there is no need to explicitly parse the input or printing the output. You only need to return the expected outputs.

Although the outputs of the tasks are different their inputs are the same. The inputs are as mentioned in Table 2.

To test your implementation, you can use hmm_mini_test.py which uses the example in the recitation slide. Final grading will not be done using only those inputs; therefore, passing the given examples may not mean you will get 100 points.

The data will be given in numpy arrays. A[i,j] is the state transition probability from state i to state j. B[i,j] is the probability of observing observation j in state i. pi[i] is the probability of initial state being

i. O[t] is the tth observation, which is an index between 0 and M-1 (inclusive).

Input Symbol                        Input Name                       Input Size

A         State Transition Matrix          NxN

B          Observation Probability Matrix NxM pi Initial State Probabilities N

O                         Observation Sequence                      T

Table 2: Explanation of the input symbols in tasks. N, M, T represent number of states, number of possible observations, length of the observation sequence, respectively. (2≤N≤10, 2≤M≤10,

1≤T≤30)

2.2          Evaluation TaskFor this task, you are going to implement forward algorithm by filling the forward function in “hmm.py”. The output should be the probability of an observation sequence given A, B, pi and the calculated alpha values in a numpy array. This can be done by calculating this probability for every possible state sequence and summing them up; however, this takes exponential time. Therefore, in this homework, you are asked to implement the forward algorithm, which runs in N2T time. If your algorithm runs in exponential time, you won’t be able to get full points.

2.3          Decoding TaskSimilar to the evaluation task, you are going to implement Viterbi algorithm by filling the viterbi function in “hmm.py”. The output should be the numpy array of most likely observation sequence given A, B, pi and the calculated deltas in a numpy array. This can be done by calculating this probability for every possible state sequence then selecting the maximum; however, this takes exponential time. Therefore, in this homework, you are asked to implement the Viterbi algorithm, which runs in N2T time. If your algorithm runs in exponential time, you won’t be able to get full point.