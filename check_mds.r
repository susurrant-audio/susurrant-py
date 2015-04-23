require(ggplot2)
require(rhdf5)

fit.mds <- function(data_file) {
  df <- t(h5read(data_file, "/X"))
  mydata <- df[sample(nrow(df), 5000), ]
  d <- dist(mydata)
  fit <- cmdscale(d, eig=TRUE, k=2)
  fit
}

graph.mds <- function(data_file) {
  fit <- fit.mds(data_file)
  x <- fit$points[,1]
  y <- fit$points[,2]
  qplot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", 
       main="Metric  MDS")
}

graph.mds("../vocab/train/gfccs_sampled.h5")