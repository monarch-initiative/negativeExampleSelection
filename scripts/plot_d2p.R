library(ggplot2)
library(ggsci)



positions <- c("Second-order LINE","First-order LINE",    "DeepWalk CBOW", 
               "DeepWalk SkipGram", "Walklets CBOW", "Walklets SkipGram" )

idg_file_path = "../d2p.tsv"
df <- read.csv(idg_file_path, sep = "\t")

p_d2p <- ggplot(df,aes(x=methods, y=MCC.mean, fill=approach)) +
  geom_bar(stat="identity", position = "dodge", alpha=0.9) +
  geom_col(position = position_dodge()) +
  geom_errorbar(
    aes(ymin = MCC.mean - MCC.std, ymax = MCC.mean + MCC.std), 
    position = position_dodge2(padding = 0.5)
  ) +
  scale_x_discrete(limits = positions) +
  ylim(-1,1)+
  coord_flip() +
  scale_fill_jco() +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_text(size=14),
    axis.text.x = element_text(size=18)) +
  xlab("")
p_d2p
ggsave( "d2p_eval.pdf",p_d2p)



