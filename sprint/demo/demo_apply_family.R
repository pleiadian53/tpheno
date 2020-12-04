# reference: http://stackoverflow.com/questions/3505701/r-grouping-functions-sapply-vs-lapply-vs-apply-vs-tapply-vs-by-vs-aggrega


# Two dimensional matrix
M <- matrix(seq(1,16), 4, 4)

# apply min to/along rows
apply(M, 1, min)  
# [1] 1 2 3 4

# apply max to columns
apply(M, 2, max)
# [1]  4  8 12 16

# 3 dimensional array
M <- array( seq(32), dim = c(4,4,2))

# Apply sum across each M[*, , ] - i.e Sum across 2nd and 3rd dimension
apply(M, 1, sum)
# Result is one-dimensional
# [1] 120 128 136 144

# Apply sum across each M[*, *, ] - i.e Sum across 3rd dimension
apply(M, c(1,2), sum)
# Result is two-dimensional
#      [,1] [,2] [,3] [,4]
# [1,]   18   26   34   42
# [2,]   20   28   36   44
# [3,]   22   30   38   46
# [4,]   24   32   40   48


# [tip]
# If you want row/column means or sums for a 2D matrix, be sure to investigate the highly optimized, lightning-quick colMeans, rowMeans, colSums, rowSums.


