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
      cpu_system = cpu / cores / 100
    ) |>
    select(-cmd)

  info <- read_csv(info_fl)

  raw_data |>
    inner_join(info, by = "pid") |>
    mutate(ratio = paste0("Train:Predict = ", num_train, ":", num_pred))
}

results <- map_dfr(stubs, pull_data)


results |>
  ggplot(aes(elapsed, cpu_system)) +
  geom_line() +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1)) +
  labs(
    x = "Elapsed Time (s)",
    y = "System CPU (of 10 cores)",
  ) +
  facet_wrap(~ratio, ncol = 1, scales = "free_x")

results |>
  ggplot(aes(elapsed, mem)) +
  geom_line() +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1)) +
  labs(
    x = "Elapsed Time (s)",
    y = "System Memory"
  ) +
  facet_wrap(~ratio, ncol = 1, scales = "free_x")

results |>
  filter(vsize_extra > 0) |>
  ggplot(aes(elapsed, vsize_extra)) +
  geom_line() +
  labs(
    x = "Elapsed Time (s)",
    y = "Virtual Memory (GB)"
  ) +
  facet_wrap(~ratio, ncol = 1, scales = "free_x")
