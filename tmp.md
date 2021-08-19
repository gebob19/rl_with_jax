For features $F$ and reward $R$
$$
Fx = R \tag{DNE for non-square F} 
$$
$$
F^T F x  = F^T R \tag{will always be square}
$$
We can now solve 
$$
x  = (F^T F)^{-1} F^T R 
$$
Where x are the co-efficients to predict the reward $R$ given features $F$