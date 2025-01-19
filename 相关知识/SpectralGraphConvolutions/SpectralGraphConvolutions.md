### **谱图卷积 (Spectral Graph Convolutions)**

#### **图卷积的定义**

谱图卷积被定义为图信号 \( x \in \mathbb{R}^N \) 与滤波器 \( g_\theta \) 在图傅里叶变换下的乘积：
\[
g_\theta \ast x = Ug_\theta U^\top x
\]
- \( U \)：图拉普拉斯矩阵 \( L \) 的特征向量矩阵（傅里叶基）。
- \( U^\top x \)：图信号 \( x \) 的图傅里叶变换。
- \( g_\theta = \text{diag}(\theta) \)：对角矩阵，参数化滤波器。

---

#### **图拉普拉斯矩阵的特性**

图卷积基于图的拉普拉斯矩阵 \( L \)：
\[
L = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
\]
- \( A \)：邻接矩阵。
- \( D \)：度矩阵。
- 拉普拉斯矩阵 \( L \) 的特征分解为：
  \[
  L = U \Lambda U^\top
  \]
  - \( \Lambda \) 是特征值对角矩阵，表示图的频率信息。

---

#### **计算问题**

直接计算 \( g_\theta \ast x \) 存在两大挑战：
1. \( U \) 的计算复杂度为 \( O(N^3) \)，大规模图上不可行。
2. 滤波器 \( g_\theta \) 需要逐点定义在特征值 \( \Lambda \) 上，限制了其灵活性。

---

#### **切比雪夫多项式近似**

为解决上述问题，论文采用 **切比雪夫多项式近似**（Chebyshev Polynomial Approximation），将滤波器 \( g_\theta \) 近似为切比雪夫多项式的加权和：
\[
g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})
\]
- \( \tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I \)：归一化特征值。
- \( T_k(x) \)：第 \( k \) 阶切比雪夫多项式，递归定义为：
  \[
  T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x), \quad T_0(x) = 1, \, T_1(x) = x
  \]

近似后的卷积操作可以写为：
\[
g_\theta \ast x \approx \sum_{k=0}^K \theta_k T_k(\tilde{L})x
\]
- \( \tilde{L} = \frac{2}{\lambda_{\max}}L - I \)：归一化拉普拉斯矩阵。
- 计算复杂度降低为 \( O(|E|) \)，与图的边数线性相关。

---

#### **局部化特性**

切比雪夫多项式的近似让卷积操作可以局部化，仅影响 \( K \) 阶邻域内的节点。这种性质非常适合图上的特征传播。