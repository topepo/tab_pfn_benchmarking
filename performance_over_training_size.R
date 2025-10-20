reticulate::use_virtualenv("./tab_pfn")

# reticulate::py_config()
# reticulate::py_list_packages(envname = "tab_pfn", type = "virtualenv")

library(tidymodels)
library(TabPFN)

load("~/github/website/RData/forested_data.RData")

# ------------------------------------------------------------------------------

set.seed(3823)
mod_1 <- tab_pfn(
  class ~ .,
  data = forested_train |> dplyr::slice_sample(n = 1, by = class)
)


# ------------------------------------------------------------------------------
# Measure effect of different "training set" sizes

size_grid <- c(1, (1:15) * 100)

pred_over_size <- NULL

for (i in size_grid) {
  set.seed(3823)
  mod_i <- tab_pfn(
    class ~ .,
    data = forested_train |> dplyr::slice_sample(n = i, by = class)
  )

  pred_i <- predict(mod_i, forested_test, type = "prob") |>
    bind_cols(forested_test) |>
    select(.pred_Yes, class) |>
    mutate(training_size = i)
  pred_over_size <- bind_rows(pred_over_size, pred_i)
}
