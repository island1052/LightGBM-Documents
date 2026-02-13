# ============================================================================
# 0. 环境配置
# ============================================================================

# 0.1 读取当前位置，并将工作环境设置为当前位置（RStudio 相对路径）
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
cat("当前工作目录:", getwd(), "\n")

# 0.2 加载包与多核配置
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,        # 数据处理与可视化
  
  lightgbm,         # LightGBM 模型
  recipes,          # 特征工程
  caret,            # 传统建模框架（网格搜索）
  ParBayesianOptimization, # 贝叶斯优化
  shapviz,          # SHAP 可视化
  DALEX,            # 可解释性框架
  DALEXtra,         # DALEX 扩展
  parallel,         # 并行计算
  doParallel,       # 并行后端
  foreach,          # 并行循环
  tidymodels,       # 现代建模框架
  Matrix,           # 稀疏矩阵
  ggthemes,         # ggplot2 主题扩展
  patchwork         # 图形拼接
)

# 获取逻辑核心数，预留2个核心给系统
n_cores <- max(1, parallel::detectCores() - 2)
cat("使用核心数:", n_cores, "\n")

# 0.3 自动创建文件夹结构
dirs <- c("0Data", "1Models", "1.5Tables", "2Figs", "3Permutation", "4SHAP")
sapply(dirs, function(d) if (!dir.exists(d)) dir.create(d, recursive = TRUE))
cat("文件夹结构已就绪\n")

# 设置随机种子，确保可重复性
set.seed(42)

# ============================================================================
# 1. 数据读取与预处理
# ============================================================================

cat("\n========== 1. 数据读取与预处理 ==========\n")

# 1.1 加载 diamonds 数据集（随机抽样 50%）
data("diamonds", package = "ggplot2")
df_raw <- as_tibble(diamonds) %>%
  sample_frac(0.3)  # 保留 30% 的样本，使用文件顶部的 set.seed(42) 保证可重复性

cat("已从原始 diamonds 中随机抽取 50% 样本。\n")
cat("抽样后数据维度:", nrow(df_raw), "行 x", ncol(df_raw), "列\n")
cat("变量名:", paste(names(df_raw), collapse = ", "), "\n")

# 1.2 数据探索
cat("\n--- 数据结构 ---\n")
str(df_raw)

cat("\n--- 数据摘要 ---\n")
summary(df_raw)

# 1.3 缺失值检查
missing_summary <- df_raw %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  mutate(missing_pct = round(missing_count / nrow(df_raw) * 100, 2))

cat("\n--- 缺失值检查 ---\n")
print(missing_summary)

if (sum(missing_summary$missing_count) == 0) {
  cat("✓ 数据无缺失值\n")
} else {
  cat("⚠ 存在缺失值，将进行处理\n")
  df_raw <- df_raw %>% drop_na()
}

# 1.4 数据清洗：移除异常值（x, y, z 中存在 0 值的记录）
cat("\n--- 异常值检查 ---\n")
n_before <- nrow(df_raw)
df_clean <- df_raw %>%
  filter(x > 0, y > 0, z > 0)
n_after <- nrow(df_clean)
cat("移除 x/y/z 为 0 的记录:", n_before - n_after, "条\n")
cat("清洗后数据维度:", nrow(df_clean), "行 x", ncol(df_clean), "列\n")

# 1.5 有序分类变量的整数编码
# diamonds 中 cut, color, clarity 本身就是 ordered factor
# 我们将其转换为整数编码，保留有序信息，LightGBM 原生支持

cat("\n--- 有序分类变量编码 ---\n")

# 查看原始因子水平
cat("cut 水平:", levels(df_clean$cut), "\n")
cat("color 水平:", levels(df_clean$color), "\n")
cat("clarity 水平:", levels(df_clean$clarity), "\n")

# 整数编码：将有序因子转为整数
df_encoded <- df_clean %>%
  mutate(
    cut     = as.integer(cut),      # Fair=1, Good=2, Very Good=3, Premium=4, Ideal=5
    color   = as.integer(color),    # D=1, E=2, ..., J=7
    clarity = as.integer(clarity)   # I1=1, SI2=2, ..., IF=8
  )

cat("编码完成，数据类型:\n")
cat(paste(names(df_encoded), "->", sapply(df_encoded, class), collapse = "\n"), "\n")

# 保存清洗后的数据
write_csv(df_encoded, "0Data/diamonds_cleaned.csv")
cat("✓ 清洗后数据已保存至 0Data/diamonds_cleaned.csv\n")

# ============================================================================
# 2. 数据分割
# ============================================================================

cat("\n========== 2. 数据分割 (8:2) ==========\n")

# 使用 tidymodels 的 initial_split 进行分层抽样（按 price 分层）
split_obj <- initial_split(df_encoded, prop = 0.8, strata = price)
df_train  <- training(split_obj)
df_test   <- testing(split_obj)

cat("训练集:", nrow(df_train), "行\n")
cat("测试集:", nrow(df_test), "行\n")

# 分离特征与目标变量
target_col   <- "price"
feature_cols <- setdiff(names(df_encoded), target_col)

X_train <- df_train %>% select(all_of(feature_cols))
y_train <- df_train[[target_col]]
X_test  <- df_test %>% select(all_of(feature_cols))
y_test  <- df_test[[target_col]]

# 构建 LightGBM 数据矩阵
# 指定分类特征列（整数编码的有序变量）
cat_features <- c("cut", "color", "clarity")

dtrain <- lgb.Dataset(
  data     = as.matrix(X_train),
  label    = y_train,
  categorical_feature = cat_features
)

dtest <- lgb.Dataset(
  data      = as.matrix(X_test),
  label     = y_test,
  reference = dtrain,
  categorical_feature = cat_features
)

# 保存训练集和测试集
write_csv(df_train, "0Data/train_set.csv")
write_csv(df_test, "0Data/test_set.csv")
cat("✓ 训练集/测试集已保存至 0Data/\n")

# ============================================================================
# 3. 方案 A: 网格搜索 (Grid Search) 
# ============================================================================
# library(foreach)
# library(doParallel)
# 
# # 注册并行后端
# cl <- makeCluster(n_cores)
# registerDoParallel(cl)
# 
# # 参数网格
# param_grid <- expand.grid(
#   learning_rate  = c(0.01, 0.05, 0.1),
#   num_leaves     = c(31, 63, 127),
#   max_depth      = c(-1, 6, 10),
#   min_data_in_leaf = c(20, 50),
#   feature_fraction = c(0.8, 1.0),
#   bagging_fraction = c(0.8, 1.0),
#   stringsAsFactors = FALSE
# )
# param_sample <- param_grid
# 
# # ============= 方案B：预先构造lgb.Dataset =============
# # 1. 创建Dataset对象
# dtrain <- lgb.Dataset(
#   data = as.matrix(X_train),
#   label = y_train,
#   categorical_feature = cat_features,
#   free_raw_data = FALSE  # 必须设置为FALSE，保留数据在内存中
# )
# 
# # 2. 预先构造Dataset（关键步骤！）
# lgb.Dataset.construct(dtrain)
# 
# # 3. 将构造好的Dataset序列化并导出到worker
# #    lightgbm的Dataset对象不能直接传递，需要保存到文件再读取
# dataset_file <- tempfile(fileext = ".bin")
# lgb.Dataset.save(dtrain, dataset_file)
# 
# # 并行网格搜索
# grid_results <- foreach(
#   i = 1:nrow(param_sample),
#   .export = c("dataset_file", "cat_features"),  # 只导出文件路径
#   .packages = c("lightgbm", "dplyr"),
#   .combine = bind_rows
# ) %dopar% {
#   # 4. 在每个worker中加载预先构造好的Dataset
#   dtrain_worker <- lgb.Dataset.load(dataset_file)
#   
#   params_i <- list(
#     objective        = "regression",
#     metric           = "rmse",
#     learning_rate    = param_sample$learning_rate[i],
#     num_leaves       = param_sample$num_leaves[i],
#     max_depth        = param_sample$max_depth[i],
#     min_data_in_leaf = param_sample$min_data_in_leaf[i],
#     feature_fraction = param_sample$feature_fraction[i],
#     bagging_fraction = param_sample$bagging_fraction[i],
#     bagging_freq     = 5,
#     verbose          = -1
#   )
#   
#   cv_result <- lgb.cv(
#     params   = params_i,
#     data     = dtrain_worker,
#     nrounds  = 500,
#     nfold    = 5,
#     early_stopping_rounds = 30,
#     verbose  = -1
#   )
#   
#   tibble(
#     learning_rate    = params_i$learning_rate,
#     num_leaves       = params_i$num_leaves,
#     max_depth        = params_i$max_depth,
#     min_data_in_leaf = params_i$min_data_in_leaf,
#     feature_fraction = params_i$feature_fraction,
#     bagging_fraction = params_i$bagging_fraction,
#     best_iter        = cv_result$best_iter,
#     best_rmse        = cv_result$best_score
#   )
# }
# 
# # 清理临时文件
# unlink(dataset_file)
# 
# # 关闭并行集群
# stopCluster(cl)
# registerDoSEQ()
# 
# # 查看效果
# cat("并行网格搜索完成！最佳RMSE:", min(grid_results$best_rmse), "\n")
# # 汇总结果
#  grid_results_df <- bind_rows(grid_results) %>%
#   arrange(best_rmse)
# 
# cat("\n--- 网格搜索 Top 5 参数组合 ---\n")
# print(head(grid_results_df, 5))
# 
# # 保存网格搜索结果
# write_csv(grid_results_df, "1.5Tables/grid_search_results.csv")
# cat("✓ 网格搜索结果已保存至 1.5Tables/grid_search_results.csv\n")
# 
# # 最优参数（网格搜索）
# best_grid <- grid_results_df %>% slice(1)
# cat("\n网格搜索最优参数:\n")
# print(best_grid)
# ============================================================================
# 3. 方案 A: 并行网格搜索 (修正版)
# ============================================================================
library(foreach)
library(doParallel)

# 注册并行后端
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# 参数网格
param_grid <- expand.grid(
  learning_rate  = c(0.01, 0.05, 0.1),
  num_leaves     = c(31, 63, 127),
  max_depth      = c(-1, 6, 10),
  min_data_in_leaf = c(20, 50),
  feature_fraction = c(0.8, 1.0),
  bagging_fraction = c(0.8, 1.0),
  stringsAsFactors = FALSE
)

# 随机抽样部分参数进行演示（全量跑太慢）
set.seed(123)
param_sample <- param_grid %>% sample_n(min(20, nrow(param_grid))) 

# --- 关键修正：导出原始数据而非 lgb.Dataset ---
# LightGBM Dataset 是 C++ 指针，不能直接跨进程传输
clusterExport(cl, varlist = c("X_train", "y_train", "cat_features"), envir = environment())
clusterEvalQ(cl, {
  library(lightgbm)
  library(dplyr)
})

cat("开始并行网格搜索...\n")

grid_results <- foreach(
  i = 1:nrow(param_sample),
  .combine = bind_rows,
  .packages = c("lightgbm", "dplyr"),
  .errorhandling = "pass" # 防止单个任务报错导致整体崩溃
) %dopar% {
  
  # 1. 在 Worker 内部构建 Dataset (这是必须的)
  # 虽然有重复构建的开销，但保证了内存安全
  dtrain_worker <- lgb.Dataset(
    data = as.matrix(X_train),
    label = y_train,
    categorical_feature = cat_features,
    free_raw_data = FALSE,
    params = list(verbose = -1)
  )
  
  # 2. 设定参数
  params_i <- list(
    objective        = "regression",
    metric           = "rmse",
    learning_rate    = param_sample$learning_rate[i],
    num_leaves       = param_sample$num_leaves[i],
    max_depth        = param_sample$max_depth[i],
    min_data_in_leaf = param_sample$min_data_in_leaf[i],
    feature_fraction = param_sample$feature_fraction[i],
    bagging_fraction = param_sample$bagging_fraction[i],
    bagging_freq     = 5,
    verbose          = -1,
    num_threads      = 1 # 关键：Worker内部强制单线程，防止CPU过载
  )
  
  # 3. 交叉验证
  cv_result <- tryCatch({
    lgb.cv(
      params   = params_i,
      data     = dtrain_worker,
      nrounds  = 1000, 
      nfold    = 3,     # 演示用3折，实际建议5折
      early_stopping_rounds = 30,
      verbose  = -1,
      stratified = FALSE # 回归任务通常设为 FALSE
    )
  }, error = function(e) return(NULL))
  
  if(is.null(cv_result)) return(NULL)
  
  # 4. 返回结果
  tibble(
    learning_rate    = params_i$learning_rate,
    num_leaves       = params_i$num_leaves,
    max_depth        = params_i$max_depth,
    best_iter        = cv_result$best_iter,
    best_rmse        = cv_result$best_score
  )
}

stopCluster(cl)
registerDoSEQ()

# 处理结果
grid_results_df <- grid_results %>% arrange(best_rmse)
cat("网格搜索最优 RMSE:", min(grid_results_df$best_rmse), "\n")

# ============= 可选的：保存结果 =============
# saveRDS(grid_results, "grid_search_results.rds")
# write.csv(grid_results, "grid_search_results.csv", row.names = FALSE)
# ============================================================================
# 4. 方案 B: 贝叶斯优化 (Bayesian Optimization)
# ============================================================================
# ============================================================================
# 4. 贝叶斯优化（完全自包含的worker节点）
# ============================================================================

lgb_bayesian_objective <- function(
    learning_rate, num_leaves, max_depth, min_data_in_leaf,
    feature_fraction, bagging_fraction, lambda_l1, lambda_l2
) {
  
  # 1. 加载包
  library(lightgbm)
  
  # 2. ⭐⭐⭐ 在worker节点重新创建数据！
  # 注意：这里需要访问X_train, y_train, cat_features
  # 这些对象必须从主节点导出
  
  dtrain_worker <- lgb.Dataset(
    data = as.matrix(X_train),
    label = y_train,
    categorical_feature = cat_features
  )
  
  # 3. 设置参数
  params <- list(
    objective        = "regression",
    metric           = "rmse",
    learning_rate    = learning_rate,
    num_leaves       = as.integer(num_leaves),
    max_depth        = as.integer(max_depth),
    min_data_in_leaf = as.integer(min_data_in_leaf),
    feature_fraction = feature_fraction,
    bagging_fraction = bagging_fraction,
    lambda_l1        = lambda_l1,
    lambda_l2        = lambda_l2,
    bagging_freq     = 5,
    verbose          = -1,
    num_threads      = 1
  )
  
  # 4. 执行CV
  cv_result <- lgb.cv(
    params   = params,
    data     = dtrain_worker,  # 使用worker自己的数据
    nrounds  = 1000,
    nfold    = 5,
    early_stopping_rounds = 50,
    verbose  = -1
  )
  
  list(Score = -cv_result$best_score, Pred = 0)
}

# ============================================================================
# 并行配置
# ============================================================================

library(parallel)
library(doParallel)
library(lightgbm)

# 设置并行集群
n_cores <- max(1, detectCores() - 2)
cl <- makeCluster(n_cores)

# ⭐⭐⭐ 关键步骤1：在worker节点加载包
clusterEvalQ(cl, {
  library(lightgbm)
  NULL
})
search_bounds <- list(
  learning_rate = c(0.01, 0.3),
  num_leaves = c(20, 100),
  max_depth = c(3, 10),
  min_data_in_leaf = c(5, 50),      # 添加这个参数！
  feature_fraction = c(0.5, 1.0),    # 添加这个参数！
  bagging_fraction = c(0.5, 1.0),    # 添加这个参数！
  lambda_l1 = c(0, 1),              # 添加这个参数！
  lambda_l2 = c(0, 1)               # 添加这个参数！
)
# ⭐⭐⭐ 关键步骤2：把数据导出到worker节点
clusterExport(cl, 
              c("X_train", "y_train", "cat_features" , "search_bounds"),  # 注意：是dtrain的原材料，不是dtrain本身！
              envir = environment())

registerDoParallel(cl)

# 执行贝叶斯优化
set.seed(42)
bayes_result <- bayesOpt(
  FUN       = lgb_bayesian_objective,
  bounds    = search_bounds,
  initPoints = 10,
  iters.n   = 20,
  iters.k   = 5,
  parallel  = TRUE,
  acq       = "ucb",
  kappa     = 2.576,
  verbose   = 1
)

# 清理
stopCluster(cl)
registerDoSEQ()
cat("\n--- 贝叶斯优化最优参数 ---\n")
best_bayes_params <- getBestPars(bayes_result)
print(best_bayes_params)

# 保存贝叶斯优化历史
bayes_history <- bayes_result$scoreSummary %>%
  as_tibble()
write_csv(bayes_history, "1.5Tables/bayesian_optimization_history.csv")
cat("✓ 贝叶斯优化历史已保存至 1.5Tables/bayesian_optimization_history.csv\n")


# ============================================================================
# 6. 使用最优参数训练最终模型
# ============================================================================


cat("\n========== 6. 训练最终模型 ==========\n")

# 6.1 从网格搜索和贝叶斯优化中各选出最佳模型
cat("从两种优化方法中筛选冠军模型...\n")

# 网格搜索的最佳参数
best_grid_params <- grid_results %>% slice(1)
grid_rmse <- best_grid_params$best_rmse

# 贝叶斯优化的最佳参数
bayes_rmse <- bayes_result$scoreSummary %>%
  as_tibble() %>%
  filter(Score == max(Score)) %>%
  slice(1) %>%
  pull(Score) %>%
  abs()  # 贝叶斯优化存储的是负值

cat("\n网格搜索最佳 RMSE:", grid_rmse, "\n")
cat("贝叶斯优化最佳 RMSE:", bayes_rmse, "\n")

# 选择两者中更优的（RMSE 更小）
if (grid_rmse <= bayes_rmse) {
  cat("\n🏆 冠军模型: 网格搜索\n")
  final_params <- list(
    objective        = "regression",
    metric           = "rmse",
    learning_rate    = best_grid_params$learning_rate,
    num_leaves       = as.integer(best_grid_params$num_leaves),
    max_depth        = as.integer(best_grid_params$max_depth),
    min_data_in_leaf = as.integer(best_grid_params$min_data_in_leaf),
    feature_fraction = best_grid_params$feature_fraction,
    bagging_fraction = best_grid_params$bagging_fraction,
    bagging_freq     = 5,
    verbose          = -1
  )
  champion_source <- "Grid Search"
} else {
  cat("\n🏆 冠军模型: 贝叶斯优化\n")
  final_params <- list(
    objective        = "regression",
    metric           = "rmse",
    learning_rate    = best_bayes_params$learning_rate,
    num_leaves       = as.integer(best_bayes_params$num_leaves),
    max_depth        = as.integer(best_bayes_params$max_depth),
    min_data_in_leaf = as.integer(best_bayes_params$min_data_in_leaf),
    feature_fraction = best_bayes_params$feature_fraction,
    bagging_fraction = best_bayes_params$bagging_fraction,
    bagging_freq     = 5,
    lambda_l1        = best_bayes_params$lambda_l1,
    lambda_l2        = best_bayes_params$lambda_l2,
    verbose          = -1
  )
  champion_source <- "Bayesian Optimization"
}

cat("\n冠军模型参数:\n")
print(final_params)


cat("最终参数:\n")
print(final_params)

# 6.1 交叉验证确定最优迭代轮数
cat("\n确定最优迭代轮数...\n")
cv_final <- lgb.cv(
  params   = final_params,
  data     = dtrain,
  nrounds  = 2000,
  nfold    = 5,
  early_stopping_rounds = 100,
  verbose  = -1
)

best_nrounds <- cv_final$best_iter
cat("最优迭代轮数:", best_nrounds, "\n")
cat("CV 最优 RMSE:", cv_final$best_score, "\n")

# 6.2 使用全部训练数据训练最终模型
final_model <- lgb.train(
  params  = final_params,
  data    = dtrain,
  nrounds = best_nrounds,
  verbose = -1
)

# 6.3 保存模型
lgb.save(final_model, "1Models/lightgbm_final_model.txt")
cat("✓ 最终模型已保存至 1Models/lightgbm_final_model.txt\n")

# ============================================================================
# 7. 模型预测与评估
# ============================================================================

cat("\n========== 7. 模型预测与评估 ==========\n")

# 7.1 在测试集上进行预测
y_pred <- predict(final_model, as.matrix(X_test))

# 7.2 计算评估指标
calc_metrics <- function(actual, predicted) {
  rmse_val <- sqrt(mean((actual - predicted)^2))
  mae_val  <- mean(abs(actual - predicted))
  ss_res   <- sum((actual - predicted)^2)
  ss_tot   <- sum((actual - mean(actual))^2)
  r2_val   <- 1 - ss_res / ss_tot
  mape_val <- mean(abs((actual - predicted) / actual)) * 100
  
  tibble(
    Metric = c("RMSE", "MAE", "R_squared", "MAPE"),
    Value  = c(rmse_val, mae_val, r2_val, mape_val)
  )
}

metrics_df <- calc_metrics(y_test, y_pred)
cat("\n--- 模型评估指标 ---\n")
print(metrics_df)

# 7.3 保存预测结果
prediction_df <- tibble(
  actual    = y_test,
  predicted = y_pred,
  residual  = y_test - y_pred
) %>%
  bind_cols(X_test)  # 附加特征信息

write_csv(prediction_df, "1.5Tables/prediction_results.csv")
cat("✓ 预测结果已保存至 1.5Tables/prediction_results.csv\n")

# 7.4 保存评估指标
write_csv(metrics_df, "1.5Tables/model_metrics.csv")
cat("✓ 评估指标已保存至 1.5Tables/model_metrics.csv\n")

# 7.5 保存特征重要性
importance_df <- lgb.importance(final_model) %>%
  as_tibble()
write_csv(importance_df, "1.5Tables/feature_importance.csv")
cat("✓ 特征重要性已保存至 1.5Tables/feature_importance.csv\n")

cat("\n特征重要性排名:\n")
print(importance_df)

# ============================================================================
# 8. 绘图模块（从 CSV 文件读取数据进行绘图）
# ============================================================================

cat("\n========== 8. 绘图模块 ==========\n")

# ---- 统一主题设置 ----
my_theme <- theme_few(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, color = "grey40"),
    axis.title    = element_text(face = "bold"),
    legend.position = "bottom"
  )

# ---- 8.1 从 CSV 读取数据 ----
cat("从 CSV 文件读取数据进行绘图...\n")

importance_plot_df  <- read_csv("1.5Tables/feature_importance.csv", show_col_types = FALSE)
prediction_plot_df  <- read_csv("1.5Tables/prediction_results.csv", show_col_types = FALSE)
metrics_plot_df     <- read_csv("1.5Tables/model_metrics.csv", show_col_types = FALSE)

# ---- 8.2 特征重要性图 ----
cat("绘制特征重要性图...\n")

p_importance <- importance_plot_df %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(x = Gain, y = Feature, fill = Gain)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  scale_fill_gradient(low = "#6BAED6", high = "#08519C") +
  labs(
    title    = "LightGBM 特征重要性 (Gain)",
    subtitle = "基于信息增益的特征排名",
    x = "信息增益 (Gain)",
    y = "特征"
  ) +
  my_theme

ggsave("2Figs/01_feature_importance.png", p_importance, 
       width = 8, height = 6, dpi = 300)
cat("✓ 特征重要性图已保存至 2Figs/01_feature_importance.png\n")

# 额外：多维度特征重要性图（Gain + Cover + Frequency）
p_importance_multi <- importance_plot_df %>%
  pivot_longer(cols = c(Gain, Cover, Frequency), 
               names_to = "Metric", values_to = "Value") %>%
  mutate(Feature = fct_reorder(Feature, Value, .fun = max)) %>%
  ggplot(aes(x = Value, y = Feature, fill = Metric)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_brewer(palette = "Set2") +
  facet_wrap(~Metric, scales = "free_x") +
  labs(
    title    = "LightGBM 多维度特征重要性",
    subtitle = "Gain / Cover / Frequency 三维度对比",
    x = "重要性数值",
    y = "特征"
  ) +
  my_theme

ggsave("2Figs/02_feature_importance_multi.png", p_importance_multi, 
       width = 12, height = 6, dpi = 300)
cat("✓ 多维度特征重要性图已保存至 2Figs/02_feature_importance_multi.png\n")

# ---- 8.3 预测值 vs 真实值散点图 ----
cat("绘制预测值 vs 真实值散点图...\n")

# 读取评估指标用于标注
rmse_val <- metrics_plot_df %>% filter(Metric == "RMSE") %>% pull(Value)
r2_val   <- metrics_plot_df %>% filter(Metric == "R_squared") %>% pull(Value)
mae_val  <- metrics_plot_df %>% filter(Metric == "MAE") %>% pull(Value)

annotation_text <- sprintf("RMSE = %.2f\nMAE = %.2f\nR² = %.4f", rmse_val, mae_val, r2_val)

p_actual_vs_pred <- prediction_plot_df %>%
  ggplot(aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.15, color = "#2171B5", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = min(prediction_plot_df$actual) + 500, 
           y = max(prediction_plot_df$predicted) - 1000,
           label = annotation_text, hjust = 0, size = 4, fontface = "bold",
           color = "darkred") +
  labs(
    title    = "预测值 vs 真实值",
    subtitle = "红色虚线为完美预测参考线",
    x = "真实价格 (Actual Price)",
    y = "预测价格 (Predicted Price)"
  ) +
  coord_equal() +
  my_theme

ggsave("2Figs/03_actual_vs_predicted.png", p_actual_vs_pred, 
       width = 8, height = 8, dpi = 300)
cat("✓ 预测值 vs 真实值散点图已保存至 2Figs/03_actual_vs_predicted.png\n")

# ---- 8.4 残差分布图 ----
cat("绘制残差分布图...\n")

# 残差直方图
p_residual_hist <- prediction_plot_df %>%
  ggplot(aes(x = residual)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60, 
                 fill = "#6BAED6", color = "white", alpha = 0.8) +
  geom_density(color = "#08519C", linewidth = 1) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(
    title    = "残差分布直方图",
    subtitle = sprintf("残差均值 = %.2f, 标准差 = %.2f", 
                       mean(prediction_plot_df$residual), 
                       sd(prediction_plot_df$residual)),
    x = "残差 (Actual - Predicted)",
    y = "密度"
  ) +
  my_theme

# 残差 vs 预测值散点图
p_residual_scatter <- prediction_plot_df %>%
  ggplot(aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.15, color = "#2171B5", size = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE, color = "#E6550D", linewidth = 1) +
  labs(
    title    = "残差 vs 预测值",
    subtitle = "检查异方差性和系统性偏差",
    x = "预测价格 (Predicted Price)",
    y = "残差 (Residual)"
  ) +
  my_theme

# 合并残差图
p_residual_combined <- p_residual_hist + p_residual_scatter +
  plot_annotation(
    title    = "残差诊断",
    theme    = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )

ggsave("2Figs/04_residual_analysis.png", p_residual_combined, 
       width = 14, height = 6, dpi = 300)
cat("✓ 残差分析图已保存至 2Figs/04_residual_analysis.png\n")

# ---- 8.5 QQ 图 ----
p_qq <- prediction_plot_df %>%
  ggplot(aes(sample = residual)) +
  stat_qq(alpha = 0.3, color = "#2171B5") +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "残差 Q-Q 图",
    subtitle = "检验残差正态性",
    x = "理论分位数",
    y = "样本分位数"
  ) +
  my_theme

ggsave("2Figs/05_residual_qq_plot.png", p_qq, 
       width = 7, height = 7, dpi = 300)
cat("✓ QQ 图已保存至 2Figs/05_residual_qq_plot.png\n")

# ============================================================================
# 9. Permutation Importance (置换重要性分析)
# ============================================================================

cat("\n========== 9. Permutation Importance ==========\n")
# 9.1 使用 DALEX 创建 explainer
# 自定义预测函数
predict_fun <- function(model, newdata) {
  predict(model, as.matrix(newdata))
}

explainer <- explain(
  model          = final_model,
  data           = as.data.frame(X_test),
  y              = y_test,
  predict_function = predict_fun,
  label          = "LightGBM",
  verbose        = FALSE
)

# 9.2 计算置换重要性
cat("计算 Permutation Importance...\n")
perm_importance <- model_parts(
  explainer,
  loss_function = loss_root_mean_square,
  B             = 500,     # 置换次数
  type          = "difference"
)

# 9.3 保存置换重要性结果
perm_df <- perm_importance %>%
  as_tibble() %>%
  filter(variable != "_full_model_" & variable != "_baseline_") %>%
  group_by(variable) %>%
  summarise(
    mean_dropout_loss = mean(dropout_loss, na.rm = TRUE),
    sd_dropout_loss   = sd(dropout_loss, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_dropout_loss))

write_csv(perm_df, "3Permutation/permutation_importance.csv")
cat("✓ 置换重要性结果已保存至 3Permutation/permutation_importance.csv\n")
cat("\n置换重要性排名:\n")
print(perm_df)

# 9.4 保存完整的置换重要性（每次置换的结果）
perm_full_df <- perm_importance %>%
  as_tibble()
write_csv(perm_full_df, "3Permutation/permutation_importance_full.csv")

# 9.5 绘制置换重要性图（从 CSV 读取）
cat("绘制 Permutation Importance 图...\n")

perm_plot_df <- read_csv("3Permutation/permutation_importance.csv", show_col_types = FALSE)

p_perm <- perm_plot_df %>%
  mutate(variable = fct_reorder(variable, mean_dropout_loss)) %>%
  ggplot(aes(x = mean_dropout_loss, y = variable)) +
  geom_col(fill = "#FB6A4A", width = 0.7) +
  geom_errorbarh(
    aes(xmin = mean_dropout_loss - sd_dropout_loss,
        xmax = mean_dropout_loss + sd_dropout_loss),
    height = 0.3, color = "grey30"
  ) +
  labs(
    title    = "Permutation Importance",
    subtitle = "特征被置换后 RMSE 的增加量（越大越重要）",
    x = "RMSE 增加量 (Dropout Loss Difference)",
    y = "特征"
  ) +
  my_theme

ggsave("3Permutation/06_permutation_importance.png", p_perm, 
       width = 8, height = 6, dpi = 300)
cat("✓ Permutation Importance 图已保存至 3Permutation/06_permutation_importance.png\n")

# 9.6 使用 DALEX 内置绘图（额外参考）
p_perm_dalex <- plot(perm_importance) +
  labs(title = "Permutation Importance (DALEX)") +
  my_theme

ggsave("3Permutation/07_permutation_importance_dalex.png", p_perm_dalex, 
       width = 8, height = 6, dpi = 300)
cat("✓ DALEX Permutation 图已保存\n")
# ============================================================================
# 9.7 置换重要性分布直方图（新增）
# ============================================================================

cat("绘制置换重要性分布直方图...\n")

# 从完整置换结果中计算分布
perm_full_df <- read_csv("3Permutation/permutation_importance_full.csv", 
                          show_col_types = FALSE)

# 计算统计信息
perm_stats <- perm_full_df %>%
  filter(variable != "_full_model_" & variable != "_baseline_") %>%
  group_by(variable) %>%
  summarise(
    mean_loss = mean(dropout_loss, na.rm = TRUE),
    sd_loss   = sd(dropout_loss, na.rm = TRUE),
    min_loss  = min(dropout_loss, na.rm = TRUE),
    max_loss  = max(dropout_loss, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_loss))

# 选择 Top 3 特征分别绘制分布
top_3_features <- perm_stats %>% slice(1:3) %>% pull(variable)

for (feat in top_3_features) {
  
  # 提取该特征的所有置换结果
  feat_perms <- perm_full_df %>%
    filter(variable == feat & variable != "_full_model_" & variable != "_baseline_") %>%
    pull(dropout_loss)
  
  # 计算统计量
  mean_perm <- mean(feat_perms, na.rm = TRUE)
  sd_perm   <- sd(feat_perms, na.rm = TRUE)
  
  # 计算 p-value（实际观测值相对于置换分布的排名）
  actual_loss <- perm_df %>% filter(variable == feat) %>% pull(mean_dropout_loss)
  p_val <- mean(feat_perms <= actual_loss, na.rm = TRUE)
  
  # 绘制直方图
  p_perm_dist <- tibble(dropout_loss = feat_perms) %>%
    ggplot(aes(x = dropout_loss)) +
    geom_histogram(aes(y = after_stat(density)), 
                   bins = 50, fill = "#BDBDBD", color = "white", alpha = 0.8) +
    geom_density(color = "black", linewidth = 0.8) +
    # 红线：真实观察值
    geom_vline(xintercept = actual_loss, color = "red", 
               linetype = "solid", linewidth = 1.5) +
    # 注释
    annotate("text", x = Inf, y = Inf, 
             label = sprintf("Real Accuracy: %.3f\np-value: %.4f", actual_loss, p_val),
             hjust = 1.05, vjust = 1.1, 
             fontface = "bold", color = "darkred", size = 4) +
    labs(
      title    = sprintf("Permutation Importance Distribution: %s", feat),
      subtitle = "直方图 = 500 次置换的分布，红线 = 实际观测值",
      x = "Dropout Loss (RMSE increase)",
      y = "Density"
    ) +
    my_theme +
    theme(plot.subtitle = element_text(size = 11, color = "grey30"))
  
  # 保存图片
  fname <- sprintf("3Permutation/08_perm_distribution_%s.png", feat)
  ggsave(fname, p_perm_dist, width = 8, height = 6, dpi = 300)
  cat(sprintf("  ✓ %s 置换分布图已保存\n", feat))
}

# 绘制所有特征的分布对比（小提琴图）
p_perm_violin <- perm_full_df %>%
  filter(variable != "_full_model_" & variable != "_baseline_") %>%
  mutate(variable = fct_reorder(variable, dropout_loss, .fun = median)) %>%
  ggplot(aes(x = variable, y = dropout_loss, fill = variable)) +
  geom_violin(alpha = 0.6, show.legend = FALSE) +
  geom_boxplot(width = 0.1, color = "black", fill = "white") +
  geom_point(data = perm_df, 
             aes(x = variable, y = mean_dropout_loss),
             color = "red", size = 3, shape = 4) +
  labs(
    title    = "Permutation Importance Distribution (All Features)",
    subtitle = "红叉 = 平均置换重要性，小提琴 = 分布",
    x = "特征",
    y = "Dropout Loss"
  ) +
  coord_flip() +
  my_theme

ggsave("3Permutation/09_perm_distributions_violin.png", p_perm_violin, 
       width = 8, height = 6, dpi = 300)
cat("✓ 置换重要性小提琴图已保存\n")
# ============================================================================
# 10. SHAP 分析 (SHapley Additive exPlanations)
# ============================================================================

cat("\n========== 10. SHAP 分析 ==========\n")

# 10.1 计算 SHAP 值
# 使用 shapviz 包，直接调用 LightGBM 的 SHAP 计算
cat("计算 SHAP 值...\n")

# 抽取测试集的子样本进行 SHAP 分析（全量计算可能较慢）
set.seed(42)
shap_sample_size <- min(2000, nrow(X_test))
shap_idx <- sample(seq_len(nrow(X_test)), shap_sample_size)
X_shap   <- as.matrix(X_test[shap_idx, ])

# 使用 shapviz 从 LightGBM 模型提取 SHAP 值
shp <- shapviz(final_model, X_pred = X_shap, X = X_shap)

cat("SHAP 值计算完成，样本数:", shap_sample_size, "\n")

# 10.2 保存 SHAP 值矩阵
shap_values_df <- as_tibble(shp$S) %>%
  mutate(sample_id = shap_idx)
write_csv(shap_values_df, "4SHAP/shap_values.csv")
cat("✓ SHAP 值矩阵已保存至 4SHAP/shap_values.csv\n")

# 保存对应的特征值
shap_features_df <- as_tibble(shp$X) %>%
  mutate(sample_id = shap_idx)
write_csv(shap_features_df, "4SHAP/shap_feature_values.csv")
cat("✓ SHAP 对应特征值已保存至 4SHAP/shap_feature_values.csv\n")

# 10.3 SHAP Summary Plot (蜂巢图 / Beeswarm Plot)
cat("绘制 SHAP Summary Plot...\n")

p_shap_summary <- sv_importance(shp, kind = "beeswarm", show_numbers = TRUE) +
  labs(
    title    = "SHAP Summary Plot (Beeswarm)",
    subtitle = "每个点代表一个样本，颜色表示特征值的高低"
  ) +
  my_theme +
  theme(legend.position = "right")

ggsave("4SHAP/08_shap_summary_beeswarm.png", p_shap_summary, 
       width = 10, height = 7, dpi = 300)
cat("✓ SHAP Summary Beeswarm 图已保存\n")

# 10.4 SHAP 特征重要性柱状图（基于平均绝对 SHAP 值）
p_shap_bar <- sv_importance(shp, kind = "bar", show_numbers = TRUE) +
  labs(
    title    = "SHAP 特征重要性 (Mean |SHAP|)",
    subtitle = "基于平均绝对 SHAP 值的特征排名"
  ) +
  my_theme

ggsave("4SHAP/09_shap_importance_bar.png", p_shap_bar, 
       width = 8, height = 6, dpi = 300)
cat("✓ SHAP 特征重要性柱状图已保存\n")

# 10.5 SHAP Dependence Plots (依赖图) — 对 Top 4 特征绘制
cat("绘制 SHAP Dependence Plots...\n")

# 获取特征重要性排名
shap_mean_abs <- colMeans(abs(shp$S))
top_features  <- names(sort(shap_mean_abs, decreasing = TRUE))[1:4]

cat("Top 4 特征:", paste(top_features, collapse = ", "), "\n")

# 为每个 Top 特征绘制依赖图
for (feat in top_features) {
  p_dep <- sv_dependence(shp, v = feat, color_var = "auto") +
    labs(
      title    = sprintf("SHAP Dependence Plot: %s", feat),
      subtitle = "颜色表示交互特征的取值"
    ) +
    my_theme +
    theme(legend.position = "right")
  
  fname <- sprintf("4SHAP/10_shap_dependence_%s.png", feat)
  ggsave(fname, p_dep, width = 8, height = 6, dpi = 300)
  cat(sprintf("  ✓ %s 依赖图已保存\n", feat))
}

# 10.6 SHAP Force Plot — 单个样本解释
cat("绘制 SHAP Waterfall Plot (单样本解释)...\n")

# 选择第一个样本进行解释
p_waterfall <- sv_waterfall(shp, row_id = 1) +
  labs(
    title    = "SHAP Waterfall Plot (单样本解释)",
    subtitle = sprintf("样本 #%d 的价格预测分解", shap_idx[1])
  ) +
  my_theme

ggsave("4SHAP/11_shap_waterfall_sample1.png", p_waterfall, 
       width = 10, height = 7, dpi = 300)
cat("✓ SHAP Waterfall 图已保存\n")

# 10.7 SHAP Force Plot — 多个样本
p_force <- sv_force(shp, row_id = 1:5) +
  labs(title = "SHAP Force Plot (前5个样本)") +
  my_theme

ggsave("4SHAP/12_shap_force_top5.png", p_force, 
       width = 14, height = 8, dpi = 300)
cat("✓ SHAP Force Plot (多样本) 已保存\n")


# ============================================================================
# 11. 综合对比：三种重要性方法对比
# ============================================================================

cat("\n========== 11. 特征重要性综合对比 ==========\n")

# 11.1 从 CSV 读取各类重要性数据
feat_imp_native <- read_csv("1.5Tables/feature_importance.csv", show_col_types = FALSE) %>%
  select(Feature, Gain) %>%
  rename(variable = Feature, value = Gain) %>%
  mutate(method = "Native (Gain)", value = value / max(value))  # 归一化

feat_imp_perm <- read_csv("3Permutation/permutation_importance.csv", show_col_types = FALSE) %>%
  select(variable, mean_dropout_loss) %>%
  rename(value = mean_dropout_loss) %>%
  mutate(method = "Permutation", value = value / max(value))

shap_vals_csv <- read_csv("4SHAP/shap_values.csv", show_col_types = FALSE) %>%
  select(-sample_id) %>%
  summarise(across(everything(), ~ mean(abs(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  mutate(method = "SHAP (Mean |SHAP|)", value = value / max(value))

# 合并
importance_comparison <- bind_rows(feat_imp_native, feat_imp_perm, shap_vals_csv)

write_csv(importance_comparison, "1.5Tables/importance_comparison.csv")

# 11.2 绘制对比图
p_compare <- importance_comparison %>%
  mutate(variable = fct_reorder(variable, value, .fun = max)) %>%
  ggplot(aes(x = value, y = variable, fill = method)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title    = "特征重要性综合对比",
    subtitle = "Native Gain / Permutation / SHAP 三种方法归一化对比",
    x = "归一化重要性",
    y = "特征",
    fill = "方法"
  ) +
  my_theme +
  theme(legend.position = "bottom")

ggsave("2Figs/14_importance_comparison.png", p_compare, 
       width = 10, height = 7, dpi = 300)
cat("✓ 特征重要性综合对比图已保存\n")

