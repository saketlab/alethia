# alethiaR

R implementation of [alethia](https://github.com/saketlab/alethia) for entity
matching, and label-free embedding-model assessment.


## Install

```r
# install.packages("remotes")
remotes::install_github("saketlab/alethia", subdir = "R")
```

## Use

Match messy strings against a canonical list:

```r
library(alethiaR)

alethia(
  c("Bombay", "Calcutta", "Madras"),
  c("Mumbai", "Kolkata", "Chennai")
)
#>   given_entity alethia_prediction alethia_score alethia_method
#> 1       Bombay             Mumbai     0.5000000      rapidfuzz
#> 2     Calcutta            Kolkata     0.4000000      rapidfuzz
#> 3       Madras             Mumbai     0.3333333      rapidfuzz
```


```r
install.packages(c("onnxr", "tok", "hfhub"))
onnxr::onnx_install()   # one-time ONNX Runtime download

embedder <- onnx_embedder("sentence-transformers/all-MiniLM-L6-v2")
alethia(c("Madras"), c("Mumbai", "Kolkata", "Chennai"), model = embedder)
#>   given_entity alethia_prediction alethia_score
#> 1       Madras            Chennai     0.7284534
#>                           alethia_method
#> 1 sentence-transformers/all-MiniLM-L6-v2
```


```r
use_onnx_backend(threads = 4)
alethia(queries, references, model = "sentence-transformers/all-MiniLM-L6-v2")
```

Any function from a character vector to a matrix is also accepted, so an API client or
a precomputed matrix works the same way:

```r
alethia(queries, references, model = function(texts) my_embeddings[texts, ])
```

Group duplicates when there is no reference list:

```r
res <- cluster_entities(c("Reebok Intl", "Reebok International", "adidas"), embedder)
cluster_records(res)
```

Score candidate models on your own data, without labels:

```r
ref <- encode(embedder, references)
qry <- encode(embedder, queries)

reference_separability(ref)   # are distinct references kept apart?
retrieval_margin(qry, ref)    # does the top match beat the runner-up decisively?
hubness(qry, ref)             # do a few references absorb everything?
```

