# Sentimen Analysis Using NaIve Bayes Binary Classifire

Here I share anther small project, implementing the NaIve Bayes Binary Classifire on a data set doing sentiment analysis.

# Data-saet
This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification.

The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg). We also include an additional 50,000 unlabeled
documents for unsupervised learning. 

In the entire collection, no more than 30 reviews are allowed for any
given movie because reviews for the same movie tend to have correlated
ratings. Further, the train and test sets contain a disjoint set of
movies, so no significant performance is obtained by memorizing
movie-unique terms and their associated with observed labels.  In the
labeled train/test sets, a negative review has a score <= 4 out of 10,
and a positive review has a score >= 7 out of 10. Thus reviews with
more neutral ratings are not included in the train/test sets. In the
unsupervised set, reviews of any rating are included and there are an
even number of reviews > 5 and <= 5.

# What is Sentiment Analysis?
Sentiment analysis involves determining whether a piece of text is positive, negative,
or neutral. This technique, also known as sentiment mining, aims to analyze
people’s opinions to aid business growth. It examines not only polarity (positive,
negative, and neutral) but also emotions such as happiness, sadness, and anger.
Various Natural Language Processing (NLP) algorithms, including rule-based, automatic, 
and hybrid methods, are employed in sentiment analysis.
For example, if a company wants to assess whether a product meets customer
needs or if there is a demand for it in the market, sentiment analysis can be applied
to the product reviews. This method is particularly effective for handling large
sets of unstructured data, enabling automatic tagging and classification. Additionally,
sentiment analysis is widely used with Net Promoter Score (NPS) surveys to
understand customer perceptions of a product or service. Its ability to process extensive
volumes of NPS responses quickly and consistently has contributed to its
popularity.

# Why is Sentiment Analysis Important?
Sentiment analysis involves interpreting the contextual meaning of words to gauge
the social sentiment surrounding a brand. It helps businesses determine if their
product will meet market demand.
According to a survey, 80% of global data is unstructured, necessitating analysis 
and organization, whether it’s in emails, texts, documents, articles, and more.
Sentiment analysis is essential because it stores data efficiently and cost-effectively,
addresses real-time issues, and aids in solving immediate scenarios.

# Sentiment Analysis Vs Semantic Analysis
# Sentiment Analysis
Sentiment analysis is concerned with identifying the emotional tone in a piece
of text. Its main objective is to classify sentiment as positive, negative, or neutral,
making it particularly useful for understanding customer opinions, reviews, and
social media comments. Sentiment analysis algorithms examine the language to
determine the prevailing sentiment, helping to gauge public or individual reactions
to products, services, or events.

# Semantic Analysis
Semantic analysis, in contrast, aims to understand the meaning and context of
the text beyond mere sentiment. It seeks to grasp the relationships between words,
phrases, and concepts within a piece of content. Semantic analysis focuses on
the underlying meaning, intent, and the connections between different elements
in a sentence. This is essential for tasks such as question answering, language
translation, and content summarization, where a deeper comprehension of context
and semantics is necessary.

# Train and Test set Distribution

In the code I have seperated the data as you can see below.
![Screenshot from 2024-06-12 19-50-37](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/3c06e9d3-d317-4f93-9f88-d3e7201ce9fc)

Understanding the class distribution in both the training and test sets is essential
for assessing the balance of the dataset and ensuring that the model is trained
31and evaluated on a representative sample of data. 
In this case, the class distribution appears to be relatively balanced, which is favorable for model training and
evaluation.

# There Are Two NaIve Bayes Used
For the futher educational purposes, I implemented the NaIve Bayes once from scratch,
and used the SKlearn's NaIve Bayes to compare.

# Self Implemented NaiveBayes

![Screenshot from 2024-06-12 19-56-35](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/3907bf21-f8ab-4d01-bbe9-8c030733e0dd)

The self-implemented Naive Bayes classifier achieved an accuracy of 86%, indicating 
that it correctly classified approximately 85% of the test samples. The
F1 score, which considers both precision and recall, is 0.8590, indicating a good
balance between precision and recall.

Looking at the confusion matrix, we can see that:

• True negatives (TN): 4333
• False positives (FP): 607
• False negatives (FN): 793
• True positives (TP): 4267

This classifier tends to have more false positives than false negatives, which means
it sometimes predicts a review to be positive when it’s actually negative more often
than vice versa. However, the number of false positives is not significantly higher
than the number of false negatives.
The sensitivity of the classifier, also known as recall or true positive rate, is
0.8432. This means that the classifier correctly identified approximately 84.32%
of the positive reviews out of all actual positive reviews in the dataset.
Overall, these metrics suggest that the self-implemented Naive Bayes classifier
performs reasonably well in sentiment analysis, with a good balance between
precision and recall and a relatively high accuracy. However, further analysis and
potential improvements could be made to reduce false positive and false negative
rates.

# SKlearn NaiveBayes

![Screenshot from 2024-06-12 19-59-24](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/0293c10a-83c0-4ba8-aca4-f368a1496ba7)

This section presents the evaluation metrics obtained from the SKlearn classifier.
The SKlearn classifier achieved an accuracy of 85.98%, which indicates
that approximately 85.98% of the test samples were classified correctly. The F1
score, which considers both precision and recall, is 0.8588. This metric represents
a balance between precision and recall, with a value close to 1 indicating excellent
performance.

The confusion matrix provides a detailed breakdown of the classifier’s performance:

• True negatives (TN): 4324
• False positives (FP): 608
• False negatives (FN): 794
• True positives (TP): 4266

The confusion matrix reveals that the SKlearn classifier exhibits a similar pattern
to the self-implemented Naive Bayes classifier, with slightly different counts
for true positives, false positives, false negatives, and true negatives.
The sensitivity of the SKlearn classifier, also known as recall or true positive
rate, is 0.8430. This metric indicates that the classifier correctly identified approximately
84.30% of the positive reviews out of all actual positive reviews in the
dataset.
Overall, the SKlearn classifier demonstrates performance comparable to the
self-implemented Naive Bayes classifier. Both models achieve similar accuracy,
34F1 score, and sensitivity, suggesting that they are equally effective in sentiment
analysis on this dataset.

# ROC Curve

The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates
the diagnostic ability of a binary classifier system as its discrimination
threshold is varied. It is created by plotting the true positive rate (sensitivity)
against the false positive rate (1 - specificity) at various threshold settings.

# AUC

The Area Under the ROC Curve (AUC) summarizes the performance of the classifier
across all possible thresholds. It provides a single scalar value that represents
the overall discriminative ability of the classifier. A higher AUC value (closer to 1)
indicates better performance, while a value of 0.5 suggests random classification
(no discrimination).

# SKlearn Model Roc Curve

![Screenshot from 2024-06-12 20-03-33](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/3db16fc5-b0bf-4e09-947d-6e0c3c4f0dd3)

• Area Under the Curve (AUC): The AUC for the SKlearn Naive Bayes
classifier is 0.93.

• Performance: This high AUC value indicates that the SKlearn classifier
has excellent performance in distinguishing between positive and negative
classes. The closer the AUC is to 1, the better the model is at making predictions.

• True Positive Rate (TPR) vs. False Positive Rate (FPR): The curve shows
a steep rise to the top-left corner, suggesting that the classifier achieves a
high true positive rate with a relatively low false positive rate initially.

# Self-Made Model Roc Curve

![Screenshot from 2024-06-12 20-04-26](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/1350b312-037b-4edb-90a7-81e010a7581b)

• Area Under the Curve (AUC): The AUC for the self-implemented Naive
Bayes classifier is 0.65.

• Performance: This lower AUC value indicates that the self-implemented
classifier has moderate performance. An AUC of 0.65 suggests that the
classifier is somewhat better than random guessing (AUC of 0.5) but still
far from ideal.

• True Positive Rate (TPR) vs. False Positive Rate (FPR): The curve rises
more gradually compared to the SKlearn classifier. This suggests that the
self-implemented classifier has a higher false positive rate for a given true
positive rate, reflecting poorer performance in distinguishing between positive
and negative instances.

# Positive Or Negative

There is also a function that prompts the user to input a review and then processes this review
through a cleaning pipeline to prepare it for sentiment analysis. Then it runes the review through
the trained model on the data set, and returns it with its binary label.

I have inputed the two reviews below and got the following results:

![Screenshot from 2024-06-12 20-11-49](https://github.com/SinaHeydari76/Sentimen-Analysis-With-NaIve-Bayes-Classifire/assets/167607101/d8d0c3b8-3b6a-40ba-bfa6-de7cb3ca6fea)
