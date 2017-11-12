### Final Review Questions

<br>1. Write a one line list comprehension to perform the following matrix
multiplication (A dot product B).  Describe how it works. This should be generalizable - do NOT hard code in the size of A and B. How would you check
it with NumPy?
```python
    A = [ [ 2, 4], [ 1, 7], [-1, 8] ]

    B = [ [3, 2, -5, 6],  [1, -3, 4, 8] ]
```

<br>2. SQL: Given table `houses` below, write a query to...

| id | sqft | beds | neighborhood | type | sale_price |
|:----------:|:------------:|:----------:|:----------:|:-----------:|:-----------:|
| 1 | 1150 | 2 | prospect-park | townhome | 244052 |
| 2 | 2600 | 3 | calhoun-isles | single_family | 609536 |
| 3 | 860 | 1 | uptown | condo | 472993 |
| 4 | 1320 | 3 | north-loop | townhome | 309485 |
| 5 | 1030 | 2 | downtown | townhome | 456141 |
| 6 | 3000 | 3 | uptown | single_family | 544431 |
| 7 | 1400 | 2 | longfellow | condo | 305314 |
| 8 | 3000 | 4 | longfellow | single_family | 485802 |
| 9 | 1700 | 3 | stephens-square | single_family | 337029 |

  * Return two statistics: the average price per bedroom, and the average price per square foot. (This should be a row with two columns - and be careful how you compute this)

  * Return the neighborhood having the highest number of single family homes per sale.

<br>3. What probability distribution would you use to describe the following situations?
  * How many customers arrive at the Starbucks in a certain time window.
  * Modeling the distribution of SAT scores (hint: treat them as continuous).
  * The number of defective parts that come off of an assembly line.

<br>4. Your friend flips a coin, then rolls dice/die, and tells you the total on the die. If the coin shows heads, she rolls one die. If it shows tails, she rolls two dice. What is the probability that the coin came up heads, given that the die/dice total is 6.   

<br>5. My company is comparing 4 different landing pages, to see which gets the best click through rate. We are a new company and so there is no baseline to compare with. The biggest difference in performance was page C over page A, with a p-value of .02. Is this significant? Page C is slightly more expensive to maintain. Should I implement Page C?

<br>6. Draw the distributions associated with a hypothesis test between two means.  Label the critical value, the Type I error, and Type II error.  Indicate on the diagram how you'd calculate the statistical power.

<br>7. What is the central limit theorem?

<br>8. In the context of machine learning, what are bias and variance?  And what is the bias-variance trade-off?

<br>9. What are the assumptions behind OLS linear regression?

<br>10. Suppose you work for company that employs a model to predict fraud. The confusion matrix for this model looks like this, where P means a fraudulent transaction, and N means a non-fraudulent transaction.

  || Predict P | Predict N |
  |----------|:---------:|:---------:|
  | Actual P |    111    |    105    |
  | Actual N |    45     |    739    |   

  Your employer says "This model has 85% accuracy! I'm only interested in letting you work on a model if you come up with a model with better accuracy than this." What do you tell him/her?  

<br>11. Are gradient descent methods deterministic? Why or why not? What can you do to increase the probability of getting optimal results?

<br>12. You are building a decision tree regressor and you split on a certain feature to get two child nodes with labels as follows:
  A: [ 92,  77,  99, 105,  99,  91, 103, 110,  88, 103]
  B: [101, 117, 117, 108, 106, 111, 103, 105, 121, 110]

  What is the information gain achieved by this split?

<br>13. What is data leakage? What are all the steps you must take when designing a model and engineering an analysis pipeline to avoid this?

<br>14. Compare and contrast random forests and boosted trees with regards to:  
  * The data that each tree in the ensemble is built on.  

  * How the quality of a split on a given feature and its value is evaluated.  
  * The general depth of each tree.

  * The bias-variance trade-off associated with each tree in the ensemble.  

  * How the ensemble can achieve a low-bias, low-variance model.  

<br>15. Compare and contrast Adaboost and gradient boosting.

<br> 16. NLP:
* What does tf-idf stand for? Why does this generally work better than other word-vectorization alternatives?
* What does it mean to lemmatize a word? Why should you do this? 

<br>17. Describe the "contents" of the matrices that come out of SVD decomposition, both their sizes and what they represent.

<br>18. Whiteboard the algorithm (psuedocode) for NMF using multiplicative updates to solve.

<br>19. What is the curse of dimensionality? What types of algorithms are affected by this? Give examples of supervised and unsupervised methods.

<br>20. Contrast collaborative vs. content based recommendation engines.

<br>21. Whiteboard the algorithm for a breadth first search in graph analysis.

<br>22. Compare and contrast Hadoop MapReduce and Spark, and give some pros and cons of each.

<br>23. Compare and contrast Spark RDDs and DataFrames. Which is faster, and why?

<br>24. What does HDFS stand for? Why should you take special consideration when writing to and reading from this file system?

<br>25. When writing software, you will often need to optimize for speed, memory, or both. Name some of the data structures in python that allow faster computation, and some of the things you can use when memory is an issue.