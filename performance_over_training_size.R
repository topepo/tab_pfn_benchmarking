library(tidymodels)
library(readr)

# ------------------------------------------------------------------------------

tidymodels_prefer()
theme_set(theme_bw())
options(pillar.advice = FALSE, pillar.min_title_chars = Inf)

# ------------------------------------------------------------------------------

all_pred <-
  read_csv("predictions_by_training_size.csv") |>
  mutate(class = factor(class, levels = c("Yes", "No")))


all_times <-
  read_csv("predictions_by_training_size_times.csv")

all_pred |>
  group_by(num_train) |>
  brier_class(class, .pred_Yes) |>
  ggplot(aes(num_train, .estimate)) +
  geom_point() +
  geom_smooth() +
  scale_x_log10()


all_pred |>
  group_by(num_train) |>
  roc_auc(class, .pred_Yes) |>
  ggplot(aes(num_train, .estimate)) +
  geom_point() +
  geom_smooth(span = .2) +
  scale_x_log10()


all_times |>
  ggplot(aes(num_train, time / 60)) +
  geom_point() +
  geom_smooth(span = .4) +
  scale_x_log10()
