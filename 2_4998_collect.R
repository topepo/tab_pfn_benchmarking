library(tidyverse)
library(lubridate)
library(janitor)

cores <- parallel::detectCores()

summary_files <- list.files(pattern = "_summary.\\csv$")
stubs <- gsub("_summary.\\csv$", "", summary_files)

pull_data <- function(stub) {
  raw_fl <- paste0(stub, ".ps.log")
  info_fl <- paste0(stub, "_summary.csv")

  raw_data <-
    read_csv(raw_fl, ) |>
    clean_names() |>
    mutate(
      vsize_gb = vsize / (1024 * 1024),
      vsize_extra = vsize_gb - vsize_gb[1],
      cpu_system = cpu / cores
    ) |>
    select(-cmd)

  info <- read_csv(info_fl)

  raw_data |> inner_join(info, by = "pid")
}
