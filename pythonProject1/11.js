graph TD
    A[Original Feature Matrix X] -->|Itemset Mining| B[Frequent Itemsets]
    B -->|Construct New Feature Matrix| C[New Feature Matrix]
    C -->|Fit Lasso Regression| D[Fitted Model]
