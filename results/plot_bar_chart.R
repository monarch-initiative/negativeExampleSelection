library(ggplot2)
library(ggsci)



positions <- c("Second-order LINE","First-order LINE",    "DeepWalk CBOW", 
  "DeepWalk SkipGram", "Walklets CBOW", "Walklets SkipGram" )

string_file_path = "string_stats.csv"
df <- read.csv(string_file_path)

p_string <- ggplot(df,aes(x=methods, y=MCC.mean, fill=approach)) +
  geom_bar(stat="identity", position = "dodge", alpha=0.9) +
  geom_col(position = position_dodge()) +
  geom_errorbar(
    aes(ymin = MCC.mean - MCC.sem, ymax = MCC.mean + MCC.sem), 
    position = position_dodge2(padding = 0.5)
  ) +
  scale_x_discrete(limits = positions) +
  ylim(0,1)+
  coord_flip() +
  scale_fill_jco() +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_text(size=14),
    axis.text.x = element_text(size=18)) +
  xlab("")
p_string
ggsave( "string_eval.pdf",p_string)


sli_file_path = "sli_stats.csv"
df <- read.csv(sli_file_path)
# FIRST WITH MODEL DANS
df_dans = df[df$model_neg_sampling=="UNS",]


plot_results <- function(df, model_neg) {
  df2 <- df[df$model_neg_sampling==model_neg,]
  print(str(df2))
  p <- ggplot(df2,aes(x=methods, y=MCC.mean, fill=approach)) +
    geom_bar(stat="identity", position = "dodge", alpha=0.9) +
    geom_col(position = position_dodge()) +
    geom_errorbar(
      aes(ymin = MCC.mean - MCC.std, ymax = MCC.mean + MCC.std), 
      position = position_dodge2(padding = 0.5)
    ) +
    scale_x_discrete(limits = positions) +
    ylim(0,1)+
    coord_flip() +
    scale_fill_jco() +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title.x = element_blank(),
      axis.text.y = element_text(size=14),
      axis.text.x = element_text(size=18)) +
    xlab("")
  return(p)
}



p_sli <- ggplot(df_dans,aes(x=methods, y=MCC.mean, fill=approach)) +
  geom_bar(stat="identity", position = "dodge", alpha=0.9) +
  geom_col(position = position_dodge()) +
  geom_errorbar(
    aes(ymin = MCC.mean - MCC.std, ymax = MCC.mean + MCC.std), 
    position = position_dodge2(padding = 0.5)
  ) +
  scale_x_discrete(limits = positions) +
  ylim(0,1)+
  coord_flip() +
  scale_fill_jco() +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_text(size=14),
    axis.text.x = element_text(size=18)) +
  xlab("")
p_sli

p_sli_dans = plot_results(df, "DANS")
p_sli_dans


ggsave( "sli_eval.pdf", p_sli)
