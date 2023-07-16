library(ggplot2)
library(ggsci)



positions <- c("HOPE","Second-order LINE","First-order LINE",    "DeepWalk CBOW", 
  "DeepWalk SkipGram", "Walklets CBOW", "Walklets SkipGram" )

string_file_path = "string_stats.csv"
df <- read.csv(string_file_path)

p_string <- ggplot(df,aes(x=methods, y=MCC.mean, fill=approach)) +
  geom_bar(stat="identity", position = "dodge", alpha=0.9) +
  scale_x_discrete(limits = positions) +
  ylim(0,1.1)+
  coord_flip() +
  scale_fill_jco() +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_text(size=14),
    axis.text.x = element_text(size=18)) +
  xlab("")
ggsave( "string_eval.pdf",p_string)


sli_file_path = "sli_stats.csv"
df <- read.csv(sli_file_path)

p_sli <- ggplot(df,aes(x=methods, y=MCC.mean, fill=approach)) +
  geom_bar(stat="identity", position = "dodge", alpha=0.9) +
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
ggsave( "sli_eval.pdf", p_sli)
