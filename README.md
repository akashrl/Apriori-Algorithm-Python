# Apriori-Algorithm-Python

Please see handout for full project details and algorithm implementation.

See ***association-rules.py*** for algorithm implementation in Python.

In this programming assignment, python was used to implement an association rule algorithm and apply it to it on the Yelp dataset. The yelp5.csv dataset was used.

## Dataset Details

yelp5.csv dataset consists of 5, 432 rows and 11 discrete attributes.

## Algorithm Details

The Apriori algorithm uses a generate-and-count strategy for deriving frequent itemsets. Candidate itemsets of size k + 1 are created by joining a pair of frequent itemsets of size k. A candidate is discarded if any one of its subsets is found to be infrequent during the pruning step.

Apriori algorithm was implemented and applied to the data. All the 11 discrete attributes were used and patterns involving all the unique attribute values were considered.

## Code Specification

Python script takes three arguments as input:

**1. trainingDataFilename:** corresponds to a subset of the Yelp data (in the same format as yelp5.csv) that is used as the training set for the algorithm.

**2. minsup:** the value of the minsup threshold used in the algorithm.

**3. minconf:** the value of the minconf threshold used in the algorithm.

The code reads in the training sets from the csv file, discovers the association rules with the provided thresholds, and then outputs the number of discovered frequent itemsets and association rules for each pattern size (starting with size of 2).

## Input and Output

The the input and output looks like this:

```
...

FREQUENT-ITEMS 2 60

FREQUENT-ITEMS 3 123

...

ASSOCIATION-RULES 2 66 ASSOCIATION-RULES 3 249 ...
```
