# rl_with_jax

$$
\theta' = \theta + \alpha F^{-1}\nabla J \\
F (\theta' - \theta) = \alpha \nabla J \\

\dfrac{1}{2} (\theta' - \theta)^T F (\theta' - \theta) = \alpha \dfrac{1}{2} (\theta' - \theta)^T \nabla J\\

\dfrac{1}{2} (\theta' - \theta)^T F (\theta' - \theta) = \alpha \dfrac{1}{2} (\theta + \alpha F^{-1}\nabla J - \theta)^T \nabla J\\

\dfrac{1}{2} (\theta' - \theta)^T F (\theta' - \theta) = \dfrac{\alpha^2}{2} \nabla_{\theta}J(\theta)^T F^{-1} \nabla_{\theta}J(\theta)\\


...\\

\alpha = \sqrt{\dfrac{2\epsilon}{\nabla_{\theta}J(\theta)^T F^{-1} \nabla_{\theta}J(\theta)}}
$$