
library(ggplot2)
library(ggsci)
library(viridis)
library(hrbrthemes)
library(tidyr)
library (dplyr)

df <- read.table("degree_only_perceptron.tsv", header = TRUE, sep="\t")
## Select just relevant columns
relevant_cols <- c('evaluation_mode','validation_unbalance_rate','matthews_correlation_coefficient','balanced_accuracy',
                  'f1_score','auroc','auprc','model_name','graph_name','holdout_number','number_of_holdouts','X.model_parameters..edge_features.',
                  'use_scale_free_distribution')
df <- df[,relevant_cols]
# rename colum
names(df)[names(df) == 'use_scale_free_distribution'] <- 'evaluation_DANS'
names(df)[names(df) == "X.model_parameters..edge_features."] <- 'edge_features'

df_summary <- df %>%
  group_by(evaluation_mode, edge_features,evaluation_DANS) %>%
  summarise( # summarize operation by group
    mean = mean(matthews_correlation_coefficient),
    std = sd(matthews_correlation_coefficient)
  )


# Change names to match acronyms in manuscript
df_summary$evaluation_DANS[df_summary$evaluation_DANS == "False"] <- 'UNS'
df_summary$evaluation_DANS[df_summary$evaluation_DANS == "True"] <- 'DANS'

df_summary$evaluation_mode <- as.factor(df_summary$evaluation_mode)


plot_bars_with_legend <- function(df, my.ylim=c(-1.0, 1.0)) {
  p<- ggplot(df, aes(x=edge_features, y=mean, fill=evaluation_DANS)) + 
    geom_bar(stat="identity", color="black", position=position_dodge()) + 
    geom_errorbar(aes(ymin=mean-std, ymax=mean+std), width=0.3,
                 position=position_dodge(.9)) +
   coord_flip() +        
    theme_bw() +
    theme(axis.title=element_blank(),
          legend.title = element_blank(),
          legend.text = element_text(size = 16),
          axis.text.x =  element_text(size = 16),
          axis.text.y = element_text(size = 18)) +
    scale_fill_npg() +
    guides(fill = guide_legend(reverse = TRUE))
  p
}

p <- plot_bars_with_legend(df_summary)
p
