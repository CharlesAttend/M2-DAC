# Question 1
$$
\begin{align*}
    \frac{\partial L \circ h}{\partial x_i} (x) &= \frac{\partial }{\partial x_i} L (\begin{pmatrix}
        h(x_1) \\
        \vdots \\
        h(x_n)
    \end{pmatrix}
    ) \\
        &= \frac{\partial }{\partial x_i} L(y) \\
        &= \frac{\partial h}{\partial x_i} * \frac{\partial L}{\partial h(x_i)}(y) \\
        &= \frac{\partial h}{\partial x_i} *\frac{\partial L}{\partial y_j}(y) \\
        &= \frac{\partial h}{\partial x_i} * (\nabla L)_j
\end{align*}
$$

# Question 2