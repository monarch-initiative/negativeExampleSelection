
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(tidyr)
library (dplyr)

df <- read.table("degree_only_perceptron.tsv", header = TRUE, sep="\t")
## Select just relevant columns
relevant_cols <- c('evaluation_mode','validation_unbalance_rate','matthews_correlation_coefficient','balanced_accuracy',
                  'f1_score','auroc','auprc','model_name','graph_name','holdout_number','number_of_holdouts','features_names',
                  'use_scale_free_distribution')
df <- df[,relevant_cols]
# rename colum
names(df)[names(df) == 'use_scale_free_distribution'] <- 'evaluation_DANS'

df_summary <- df %>%
  group_by(evaluation_mode, evaluation_DANS) %>%
  summarise( # summarize operation by group
    mean = mean(matthews_correlation_coefficient),
    std = sd(matthews_correlation_coefficient)
  )

