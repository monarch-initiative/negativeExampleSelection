library(ggplot2)
library(ggsci)
library(viridis)
library(hrbrthemes)
library(tidyr)
library(dplyr)

sli_file_path = "sli_results.csv"
string_file_path = "string_results.csv"

prepare_dataset <- function(file_path) {
  df <- read.table(sli_file_path, header = TRUE, sep=",")
  # filter to include DANS negative sampling in the perceptron phase
  df <- df[df$model_negative_examples=='DANS',]
  # filter to include graph ML methods
  df <-  df[which(df$features_names %in% c("DeepWalk CBOW","DeepWalk GloVe","DeepWalk SkipGram",
                                           "First-order LINE","HOPE","Second-order LINE",
                                           "Walklets CBOW", "Walklets GloVe","Walklets SkipGram")),]
  return (df)
}


summarise_df <- function(df) {
  df_summary <- df %>% 
    group_by(evaluation_mode, evaluation_negative_sampling_method, features_names) %>% 
    summarise(mean.auprc = mean(auprc),  sd.auprc = sd(auprc), mean.auroc=mean(auroc), sd.auroc=sd(auroc),
              mean.mcc=mean(matthews_correlation_coefficient), sd.mcc=sd(matthews_correlation_coefficient),
              mean.f1=mean(f1_score), sd.f1=sd(f1_score),
              mean.ba=mean(balanced_accuracy), sd.ba=sd(balanced_accuracy))
  return (df_summary)
}


plot_summary <- function (summary_df) {
  p <- summary_df %>%
    ggplot( aes(x=features_names, y=mean.mcc, fill=evaluation_negative_sampling_method)) +
    geom_boxplot() +
    scale_fill_viridis(discrete = TRUE, alpha=0.6) +
   # geom_jitter(color="black", size=0.4, alpha=0.9) +
    theme_ipsum() +
    theme(
      legend.position=c(0.1, 0.95),
      legend.title = element_blank(),
      plot.title = element_text(size=11),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
    ) +
    ggtitle("A boxplot with jitter") +
    xlab("")
  return (p)
}



sli_df = prepare_dataset(sli_file_path)
sli_summary <-summarise_df(sli_df)
p <- plot_summary(sli_summary)
p



string_df = prepare_dataset(string_file_path)
string_summary <-summarise_df(string_df)
p <- plot_summary(string_summary)
p




