#importing dataset
from sklearn.datasets import load_iris

#importing classifier (tree)
from sklearn import tree

iris = load_iris()

#print (iris.feature_names)
#print (iris.target_names)


#getting trainig data
mytraindata = iris.data #data (2D array)


target = iris.target # labels of data (2D array)


#getting test sample
#testsample = [.3, 1, 0.3, 1.7]
#testsample = iris.data[69]

#Getting test sample detsils from user
print ("Enter Sample Features:")
print ("--------------------------------------")

print ("Enter Sepal Length:")
sepal_lenth = input()

print ("Enter Sepal Width:")
sepal_width = input()

print ("Enter Petal Length:")
petal_length = input()

print ("Enter Petal Width:")
petal_width = input()

#Generating testing sample from input
testsample = [sepal_lenth, sepal_width, petal_length,petal_width]


#converting to 2D
testsample = [testsample]

myclf = tree.DecisionTreeClassifier() #defining Classifier
myclf = myclf.fit(mytraindata, target) #Training 

#Predicting for a test sample
result = myclf.predict(testsample)


#Printing result
if result == 0:
	print ("Result sample is of Iris setosa")
elif result == 1:
	print ("Result sample is of Iris versicolor")
else:
	print ("Result sample is of Iris virginica")




# For Visualizing the decision tree
# before using visualization, install graphviz first
# for windows use "Conda install graphviz" to install graphviz

import graphviz 
dot_data = tree.export_graphviz(myclf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("output_iris_graph_pdf")



  
