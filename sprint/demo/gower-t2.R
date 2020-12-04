library(cluster)  # we'll use these packages
library(fpc)

# here we're generating 45 data in 3 clusters:
set.seed(3296)    # this makes the example exactly reproducible
n      = 15
cont   = c(rnorm(n, mean=0, sd=1),
           rnorm(n, mean=1, sd=1),
           rnorm(n, mean=2, sd=1) )
bin    = c(rbinom(n, size=1, prob=.2),
           rbinom(n, size=1, prob=.5),
           rbinom(n, size=1, prob=.8) )
ord    = c(rbinom(n, size=5, prob=.2),
           rbinom(n, size=5, prob=.5),
           rbinom(n, size=5, prob=.8) )
data   = data.frame(cont=cont, bin=bin, ord=factor(ord, ordered=TRUE))

# read table 
basedir = '/Users/pleiades/Dropbox/Project-CUMC/SPRINT/sprint_pop/data'
fpath = '/Users/pleiades/Dropbox/Project-CUMC/SPRINT/sprint_pop/data/baseline.csv'
# data = read.csv(fpath, fill = TRUE, header = TRUE, sep=',', colClasses=c('NULL',rep("NA", 28)))
df = read.csv(fpath, fill = TRUE, header = TRUE, sep=',')
data = df[,2:ncol(df)] # 2: 30

# this returns the distance matrix with Gower's distance:  
g.dist = daisy(data, metric="gower", 
               list(asymm=c(1, 4, 8, 10, 13, 14, 16, 17, 18, 19, 20, 28))) # type=list(symm=2)

# write CSV 
D = as.matrix(g.dist)
# write.csv(D, file = "dissimilarity.csv", row.names = FALSE)
write.table(D, file = "dissim_baseline.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")

# we can start by searching over different numbers of clusters with PAM:
# Partitioning Around Medoids
pc = pamk(g.dist, krange=1:20, criterion="asw")
pc[2:3]

# $nc
# [1] 2                 # 2 clusters maximize the average silhouette width
# 
# $crit
# [1] 0.0000000 0.6227580 0.5593053 0.5011497 0.4294626

pc = pc$pamobject;  pc  # this is the optimal PAM clustering

# Medoids:
#      ID       
# [1,] "29" "29"
# [2,] "33" "33"
# Clustering vector:
#  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 
#  1  1  1  1  1  2  1  1  1  1  1  2  1  2  1  2  2  1  1  1  2  1  2  1  2  2 
# 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 
#  1  2  1  2  2  1  2  2  2  2  1  2  1  2  2  2  2  2  2 
# Objective function:
#     build      swap 
# 0.1500934 0.1461762 
# 
# Available components:
# [1] "medoids"    "id.med"     "clustering" "objective"  "isolation" 
# [6] "clusinfo"   "silinfo"    "diss"       "call" 

### Hierarchical Clustering ###

hc.m = hclust(g.dist, method="median")
hc.s = hclust(g.dist, method="single")
hc.c = hclust(g.dist, method="complete")
# windows(height=3.5, width=9)
# layout(matrix(1:3, nrow=1))
plot(hc.m)
plot(hc.s)
plot(hc.c)
