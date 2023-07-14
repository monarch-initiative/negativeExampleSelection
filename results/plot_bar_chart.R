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
  df_summary$evaluation <- paste0(df_summary$evaluation_negative_sampling_method, " (", df_summary$evaluation_mode, ")")
  return (df_summary)
}


plot_summary <- function (summary_df) {
  p <- summary_df %>%
    ggplot( aes(x=features_names, y=mean.mcc, fill=evaluation)) +
    geom_boxplot() +
    scale_fill_viridis(discrete = TRUE, alpha=0.75) +
   # geom_jitter(color="black", size=0.2, alpha=0.9) +
    theme_ipsum() +
    theme(
      legend.position=c(0.1, 0.95),
      legend.title = element_blank(),
      plot.title = element_text(size=11),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
    ) +
    xlab("")
  return (p)
}


plot_summary_bars <- function (summary_df) {
  df <- summary_df %>% as.data.frame()
  p <- ggplot(df) +
    geom_bar( aes(x=features_names, y=mean.mcc), stat="identity", fill="skyblue", alpha=0.7) +
    geom_errorbar( aes(x=features_names, ymin=features_names-sd.mcc, ymax=features_names+sd.mcc),
                   width=0.4, colour="orange", alpha=0.9, linewidth=1.3) +
    #scale_fill_viridis(discrete = TRUE, alpha=0.75) 
  #  theme_ipsum() +
    theme(
      legend.position=c(0.1, 0.95),
      legend.title = element_blank(),
      plot.title = element_text(size=11),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
    ) +
    xlab("")
  return (p)
}







sli_df = prepare_dataset(sli_file_path)
sli_summary <-summarise_df(sli_df)
p <- plot_summary_bars(sli_summary)
p

write.csv(sli_summary, "sli_summary.csv", row.names=FALSE)


string_df = prepare_dataset(string_file_path)
string_summary <-summarise_df(string_df)
p <- plot_summary(string_summary)
p

df <- sli_summary %>% as.data.frame()
ggplot(df) +
  geom_bar( aes(x=features_names, y=mean.mcc, fill=features_names), alpha=0.7) +
  geom_errorbar( aes(x=features_names, ymin=mean.mcc-sd.mcc, ymax=mean.mcc+sd.mcc),
                 width=0.4, colour="orange", alpha=0.9, linewidth=1.3) +
  #  theme_ipsum() +
  theme(
    legend.position=c(0.1, 0.95),
    legend.title = element_blank(),
    plot.title = element_text(size=11),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
  ) +
  xlab("")

