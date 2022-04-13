*****IMPORTANT:*****

TODO: MINOR CHANGES IN DATA EXTRACTION PIPELINE TO MAKE TASK 2 AS MULTI-LABEL CLASSIFICATION AND TASK3 TO BE TASK GIVEN BELOW.
multi-label classification: https://www.researchgate.net/publication/327110772_A_Survey_on_Multi_Label_Classification (TOREAD)
                        *****https://paperswithcode.com/task/multi-label-classification (TOREAD)*****
                        
CURRENT STRATEGIES:

1. ANCHORING (PREFERABLE BUT TIME CONSUMING TO IMPLEMENT):  Create 5 labels Disposition,Disposition,Disposition,Non-Disposition,Undetermined. Make discrimination such that given two or three attribute-labels, one of their permutation is most likely or deterministically(probability is one for that permutation and for others it is zero) appear in training set. <--- This strategy has to be thought about.
2. MULTI-LABEL CLASSIFICATION (RISKY BUT EASY TO IMPLEMENT): Create 672 classes, each class representing a 5-tuple of (Action,Certainity,Negation,Actor,TEMPORALITY) out of all possible 5-tuples. While training combine several classes as segment_sums to get the probability of a single event happening like TEMPORALITY = "PAST" occuring as a attribute. USE CROSS_ENTROPY LOSS OR DICE LOSS OR ANY OTHER MULTI_LABEL LOSS over the 5 classes (Action,Certainity,Negation,Actor,Temporality). MonteCarlo Approximation should give desired results but will the method converge to optimal behaviour is not guranteed.
