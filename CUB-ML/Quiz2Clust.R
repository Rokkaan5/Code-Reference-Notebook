##################
## Quiz 2 ML Code Support - Clustering Questions
## Gates
########################################

library(stats)  
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/dist
#install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)
library(factoextra)

setwd("C:/Users/rokka/Desktop/5622/R-code")
(AdmissionsData<-read.csv("StudentSummerProgramDataQuiz2.csv"))

## Create a dataframe from this dataset that can be used
## with kmeans and distance metrics such as Eucli Dists
## HINT:
(Num_Only_Adms_Data <- AdmissionsData[-c(1, 2, 3, 5, 8)])

## Distance Metric Matrices using dist
(Eucl_Dist <- stats::dist(Num_Only_Adms_Data,method="minkowski", p=2))  

library(stylo)
(CosSim <- stylo::dist.cosine(as.matrix(Num_Only_Adms_Data)))

Hist1 <- stats::hclust(Eucl_Dist, method="ward.D2")
plot(Hist1)
## Ward  - a general agglomerative 

##################  k - means----------------

k <- 4 # number of clusters
(kmeansResult <- stats::kmeans(Num_Only_Adms_Data, k,)) ## uses Euclidean
kmeansResult$withinss

stats::kmean()
############# To use a different sim metric----------
## one option is akmeans
## https://cran.r-project.org/web/packages/akmeans/akmeans.pdf

# library(akmeans)
# (akmeans(Num_Only_Adms_Data, min.k=3, max.k=3, verbose = TRUE))




################ Cluster vis-------------------
(factoextra::fviz_cluster(kmeansResult, data = Num_Only_Adms_Data,
                          ellipse.type = "convex",
                          #ellipse.type = "concave",
                          palette = "jco",
                          #axes = c(1, 4), # num axes = num docs (num rows)
                          ggtheme = theme_minimal()))


## Silhouette........................
factoextra::fviz_nbclust(Num_Only_Adms_Data, method = "silhouette", 
                         FUN = hcut, k.max = 5)



(df_2d <- AdmissionsData[c(4,7)])
k <- 5 # number of clusters
(kmeansResult <- stats::kmeans(df_2d, k)) ## uses Euclidean
