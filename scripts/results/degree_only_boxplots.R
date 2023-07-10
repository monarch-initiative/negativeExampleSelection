library(ggplot2)
library(ggsci)
library(viridis)
library(hrbrthemes)
library(tidyr)
library(dplyr)



extract_df <- function(original_df_filepath) {
  relevant_cols <- c('evaluation_mode','validation_unbalance_rate','matthews_correlation_coefficient','balanced_accuracy',
                     'f1_score','auroc','auprc','model_name','graph_name','holdout_number','number_of_holdouts','features_names',
                     'f1_score','auroc','auprc','model_name','graph_name','holdout_number','number_of_holdouts','X.model_parameters..edge_features.',
                     'use_scale_free_distribution')
  df <- read.table(original_df_filepath, header = TRUE, sep="\t")
  df <- df[,relevant_cols]
  # rename column
  names(df)[names(df) == 'use_scale_free_distribution'] <- 'evaluation_DANS'
  names(df)[names(df) == "X.model_parameters..edge_features."] <- 'edge_features'
  # Remove the brackets from the feature names
  df$edge_features<-gsub("\\[","",as.character(df$edge_features))
  df$edge_features<-gsub("]","",as.character(df$edge_features))
  # Change names to match acronyms in manuscript
  df$evaluation_DANS[df$evaluation_DANS == "False"] <- 'UNS'
  df$evaluation_DANS[df$evaluation_DANS == "True"] <- 'DANS'
  df$evaluation_mode <- as.factor(df$evaluation_mode)
  return(df)
}


sli_df <- extract_df("degree_only_perceptron_sli.tsv")
string_df <- extract_df("degree_only_perceptron_string.tsv")



sli_plot <- ggplot(data=sli_df, mapping=aes(x=edge_features, y=matthews_correlation_coefficient, fill=evaluation_mode))+
    geom_boxplot()

string_plot <- ggplot(data=string_df, mapping=aes(x=edge_features, y=matthews_correlation_coefficient, fill=evaluation_mode))+
  geom_boxplot()



