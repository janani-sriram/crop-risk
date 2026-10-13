suppressPackageStartupMessages({
  library(pROC)
  library(ggplot2)
})

if (!file.exists("model_predictions.csv")) {
  stop("model_predictions.csv not found! Run 'python stage1_model.py' first.")
}

data <- read.csv("model_predictions.csv")
cat("[R] Loaded model predictions from Python pipeline. Total records:", nrow(data), "\n")

se_cart  <- (data$prob_cart - data$actual_failure)^2
se_bayes <- (data$prob_bayes - data$actual_failure)^2

t_test_result <- t.test(se_cart, se_bayes, paired = TRUE, alternative = "greater")

cat("\n----- TRACK 5: HYPOTHESIS TESTING (PAIRED T-TEST) -----\n")
cat("Null Hypothesis (H0): Mean Squared Error(CART) <= Mean Squared Error(Bayesian)\n")
cat("Alternative (H1)   : Mean Squared Error(CART) > Mean Squared Error(Bayesian)\n")
cat("t-statistic        :", round(t_test_result$statistic, 4), "\n")
cat("p-value            :", format.pval(t_test_result$p.value, digits = 4), "\n")
cat("95% CI of Diff     : [", round(t_test_result$conf.int[1], 5), ",", round(t_test_result$conf.int[2], 5), "]\n")
if (t_test_result$p.value < 0.05) {
  cat("Result             : Reject H0. Bayesian model shows statistically significant error reduction (p < 0.05).\n")
} else {
  cat("Result             : Fail to reject H0.\n")
}
cat("-------------------------------------------------------\n\n")

roc_cart  <- roc(data$actual_failure, data$prob_cart, quiet = TRUE)
roc_bayes <- roc(data$actual_failure, data$prob_bayes, quiet = TRUE)

png("final_roc_comparison.png", width = 800, height = 600, res = 120)
plot(roc_bayes, col = "#2E86AB", lwd = 3, main = "Comparative ROC Curve (R Analysis of Python Models)")
lines(roc_cart, col = "#D90429", lwd = 2, lty = 2)
legend("bottomright", 
       legend = c(paste0("Bayesian Tree (AUC = ", round(auc(roc_bayes), 3), ")"),
                  paste0("Standard CART (AUC = ", round(auc(roc_cart), 3), ")")),
       col = c("#2E86AB", "#D90429"), 
       lwd = c(3, 2), 
       lty = c(1, 2))
grid()
dev.off()
cat("[R] Generated 'final_roc_comparison.png'\n")

data$Failure_Label <- factor(data$actual_failure, levels = c(0, 1), labels = c("Normal Harvest", "Yield Failure"))

p <- ggplot(data, aes(x = prob_bayes, fill = Failure_Label)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#2A9D8F", "#E76F51")) +
  labs(title = "Posterior Risk Score Density Distribution",
       subtitle = "Separation of crop failure vs. normal yield under Bayesian Decision Tree",
       x = "Predicted Failure Probability P(Y=1 | x)",
       y = "Density",
       fill = "Empirical Outcome") +
  theme_minimal(base_size = 12)

ggsave("final_risk_distribution.png", plot = p, width = 7, height = 5, dpi = 300)
cat("[R] Generated 'final_risk_distribution.png'\n")
