#!/usr/bin/python
from collections import defaultdict
import matplotlib.pyplot as mplt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import grid_search
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

poi_label = ['poi']

features_list = poi_label + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Total number of data points
print "Total number of data points: %i" %len(data_dict)

# Total number of features for each person
print "Total number of features for each person in the dataset: %i" %len(data_dict["METTS MARK"])

# Total number of POIs and non-POIs
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi += 1
    
print "Total number of poi: %i" % poi
print "Total number of non-poi: %i" % (len(data_dict) - poi)

# Create dictionary to indicate number of NaN values for each feature
nan_feature_counter = defaultdict(int)
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            nan_feature_counter[feature] += 1

# Number of NaN values for each feature
print nan_feature_counter.items()



### Task 2: Remove outliers

# Plot salary vs bonus to identify any outliers
features = ['bonus', 'salary']
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    mplt.scatter(salary, bonus)
mplt.xlabel("salary")
mplt.ylabel("bonus")
mplt.show()

# TOTAL is removed from the data set as it is the sum of all persons and heavily skews the plot.
data_dict.pop('TOTAL', 0)

# This is not related to what we are examining so it is removed.
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# Plot salary vs bonus without outlier.
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    mplt.scatter(salary, bonus)
mplt.xlabel("salary")
mplt.ylabel("bonus")
mplt.show()

# Identify the 4 outliers
outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN':
        continue
    outliers.append((key, int(value)))
    
remainingoutliers = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
print "The remaining four outliers are:"
print remainingoutliers



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Creating two variables which represent the share of TO messages to a POI and FROM
# messages from a person of POI.

for person in my_dataset:
    to_messages = my_dataset[person]['to_messages']
    from_poi_to_this_person = my_dataset[person]['from_poi_to_this_person']
    if to_messages != 'NaN' and from_poi_to_this_person != 'NaN':
        my_dataset[person]['ratio_of_messages_from_poi'] = from_poi_to_this_person/float(to_messages)
    else:
        my_dataset[person]['ratio_of_messages_from_poi'] = 'NaN'
    from_messages = my_dataset[person]['from_messages']
    from_this_person_to_poi = my_dataset[person]['from_this_person_to_poi']    
    if from_messages != 'NaN' and from_this_person_to_poi != 'NaN':
        my_dataset[person]['ratio_of_messages_to_poi'] = from_this_person_to_poi/float(from_messages)
    else:
        my_dataset[person]['ratio_of_messages_to_poi'] = 'NaN'

features_list = features_list + ['ratio_of_messages_from_poi'] + ['ratio_of_messages_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Select 6 best features using K-Best feature selection
k_best_selection = SelectKBest(k=6)
k_best_selection.fit(features, labels)

selection_results = zip(k_best_selection.get_support(), features_list[1:], k_best_selection.scores_)
selection_results = sorted(selection_results, key=lambda x: x[2], reverse=True)

print "K-best features:", selection_results

# Updates features list from K-Best

my_features = ['poi',
               'exercised_stock_options',
               'total_stock_value',
               'bonus',
               'salary',
               'ratio_of_messages_to_poi',
               'deferred_income']

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, my_features)
'''
### Using K-best = 5, precision and recall >.3.
Accuracy: 0.86050	Precision: 0.51572	Recall: 0.38550	F1: 0.44120	F2: 0.40600
Total predictions: 14000	True positives:  771	False positives:  724	False negatives: 1229	True negatives: 11276
'''

'''
### Using K-best = 5.
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2, tol=0.001)
test_classifier(clf, my_dataset, my_features)

Accuracy: 0.79636	Precision: 0.29355	Recall: 0.30250	F1: 0.29796	F2: 0.30067
Total predictions: 14000	True positives:  605	False positives: 1456	False negatives: 1395	True negatives: 10544
'''

'''
### Tuned classfier with > .3 precision, but recall <.3.
from sklearn.neighbors import KNeighborsClassifier
parameters = {'n_neighbors':(1,2,3,4,5,6,7,8,9,10), 'weights':('uniform', 'distance')}
knc = KNeighborsClassifier()
clf = grid_search.GridSearchCV(knc, parameters)
test_classifier(clf, my_dataset, my_features)

### Best evaluation score achieved at K-best = 5. When using 6 best features, evaluation scores declined.
Accuracy: 0.87079	Precision: 0.61329	Recall: 0.25850	F1: 0.36370	F2: 0.29232
Total predictions: 14000	True positives:  517	False positives:  326	False negatives: 1483	True negatives: 11674

### Tuned parameters
print clf.best_params_
{'n_neighbors': 3, 'weights': 'uniform'}
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, stratify=labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'accuracy: ', accuracy_score(pred, labels_test)
print 'precision: ', precision_score(labels_test, pred)
print 'recall: ', recall_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your resultssky.

dump_classifier_and_data(clf, my_dataset, my_features)