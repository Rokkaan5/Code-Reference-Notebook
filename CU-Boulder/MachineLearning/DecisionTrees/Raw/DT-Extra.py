## Dataset

## https://drive.google.com/file/d/1lQ85Ftir5F8gNpI833hC52_8GgcMiGJp/view?usp=share_link

## conda install python-graphviz.

from sklearn import tree
import pandas as pd

import graphviz 

## Read in data
filename = "C:/Users/profa/Desktop/UCB/ML CSCI 5622/Data/HeartRisk_JustNums_4D_Labeled.csv"
HeartRiskDF = pd.read_csv(filename)
print(HeartRiskDF)

Label = HeartRiskDF["Label"]
print(Label)

HeartRiskDF_noLabel = HeartRiskDF.drop(["Label"], axis=1)
print(HeartRiskDF_noLabel)


MyDT_Classifier = tree.DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(HeartRiskDF_noLabel, Label)

#print(HeartRiskDF_noLabel.columns)
################ Vis
#import graphviz 

 

TREE_Vis = tree.export_graphviz(MyDT_Classifier, 
                    out_file=None, 
                    feature_names=HeartRiskDF_noLabel.columns,  
                    class_names=["Risk", "NoRisk", "Medium"],  
                    filled=True, rounded=True,  
                    special_characters=True)  

graph = graphviz.Source(TREE_Vis)  
graph 


#Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_pdf("My_DT_tree.pdf")
