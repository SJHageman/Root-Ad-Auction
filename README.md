# Root Ad Auction Group Project

## Cőde Boot Camp Project
This project was part of an annual one month long coding [boot camp](https://www.erdosinstitute.org/code) that was organized by the [Erdős Institute](https://www.erdosinstitute.org).  The goal of the boot camp was to provide PhD students with foundational knowdlegde to analyze data, build and use models and simulations, and interpret their findings.  It culiminated with a group project in the final week that was presented to physics and math professors as well as industry sponsors of the event.  Our team consisted of 5 PhD students in the physics department.

## Project Description
The project that we chose to tackle was a suggested project by one of the industry sponsors of the boot camp, Root Insurance. The dataset that they provided consisted of one month of ad auction data from a real-time bidding (RTB) agent.  The idea behind an ad auction is that an auction will be held to determine which ad will be shown in an app to a particular user. So, an app will send a request for an ad to an ad exhange and then an auction will be held.  RTB agents will then submit a bid amount for the inventory, and the agent with the highest bid wins the auction and gets to serve their ad to the end user.  When this data was collected, this was a second price auction, so the winner of the auction doesn't pay the amount they bid, instead they pay the amount of the second highest bidder.  These RTB agents are just models that use features describing the inventory for which an ad has been requested and output a bid amount for the inventory based upon the likelihood that the end user will click on their ad.

The main goal for this project was to train a classifier which indentifies impressions that are going to lead to clicks on the ad.  A significant challenge for this project is the fact that this dataset is highly imbalanced (i.e. most people don't click on the ad), as well as categorical in nature.  Additionally, our group was limited to only using personal machines for this work, so the large size of the dataset meant that we had to work carfully to avoid computations taking too long or too much memory.

## Results
Since the dataset consisted of auctions that Root Insurance won, I am unable to share details or descriptors of the data that was used for the project. However, I can discuss how our model performed in classifying impressions that will lead to clicks.
### 'Click' Classifier
Most of the data provided in this dataset is categorical in nature, and this data needs to be turned into numerical values for models to train on it. There are several methods to accomplish this:

* One Hot Encoding
* Hashing
* Leave One Out
* Ordinal
* Binary
* ... and many more

The scikit-learn-contrib package [category_encoders](https://contrib.scikit-learn.org/categorical-encoding/) provides convenient transformers for many of these common methods.

The encoding method used for our model is LeaveOneOut, which is tends to work well for high cardinality categorical data.  From the [documentation](https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html):
>For the case of categorical target: features are replaced with a blend of posterior probability of the target given particular categorical value and the prior probability of the target over all the training data.

Once our categorical values are encoded, the data is then standardized such that the mean is removed and the variance is one.  This is done becuase we don't want to implicity weight our features, and by standardizing all of our features, we avoid this problem.

The model we implemented is logistic regression with stochastic gradient descent (SGD). This was done using SGDClassifier from scikit-learn, and regularization was implemented by Elastic Net.  


