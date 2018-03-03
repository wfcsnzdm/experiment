+采用三个回归算法预测软件缺陷个数，有三点需要强调：
 +1，采用out-of-sample bootstrap检验方法，参见An Empirical Comparison of Model Validation Techniques for Defect Prediction Models
 +2，采用fpa指标（参见A Learning-to-Rank Approach to Software Defect Prediction），平均绝对误差（每一个真实缺陷个数上的平均绝对误差）
 +3，进行预测时，对预测结果进行了四舍五入
