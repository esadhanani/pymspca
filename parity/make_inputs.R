# Export mtcars (base R) for the parity suite.
write.csv(as.matrix(mtcars), "parity/data/X_mtcars.csv", row.names = FALSE)
write.csv(cor(as.matrix(mtcars)), "parity/data/S_mtcarscor.csv", row.names = FALSE)
