# Algorithm Pseudocode for MetaCausalField

This document provides formal algorithmic descriptions for the key components of MetaCausalField, as presented in the paper (Section 3).

---

## Algorithm 1: End-to-End Training of MetaCausalField

```latex
\begin{algorithm}
\caption{End-to-End Training of MetaCausalField}
\begin{algorithmic}[1]
\REQUIRE Visual encoder $E_v$, dataset $\mathcal{D} = \{(x_v^{(i)}, x_l^{(i)})\}_{i=1}^N$, 
          number of propagation steps $K$, learning rate $\eta$, epochs $E$
\ENSURE Trained parameters $\theta$ for causal field $\mathcal{F}_\theta$

\STATE Initialize parameters $\theta$ randomly
\STATE Initialize optimizer with learning rate $\eta$

\FOR{epoch $e = 1$ to $E$}
    \FOR{each batch $\mathcal{B} \subset \mathcal{D}$}
        \STATE // Step 1: Extract visual features
        \STATE $Z = E_v(\{x_v \mid (x_v, x_l) \in \mathcal{B}\})$  \COMMENT{$Z \in \mathbb{R}^{B \times N \times C}$}
        
        \STATE // Step 2: Build continuous causal field
        \STATE $\mathcal{F}^0 = \text{FieldProj}(Z)$  \COMMENT{Initial field via Eq. 1}
        
        \STATE // Step 3: Compute directional influence matrix
        \STATE $G = \text{InfluenceFunc}(\mathcal{F}^0)$  \COMMENT{$G \in \mathbb{R}^{N \times N}$ via Eq. 2}
        
        \STATE // Step 4: Causal propagation (multi-step rollout)
        \STATE $\mathcal{F}^1 = G^T \mathcal{F}^0$
        \FOR{$k = 2$ to $K$}
            \STATE $\mathcal{F}^k = \alpha \cdot G^T \mathcal{F}^{k-1} + (1-\alpha) \cdot \text{InfluenceFunc}(\mathcal{F}^{k-1})$
        \ENDFOR
        
        \STATE // Step 5: Contrastive loss (implicit factor discovery)
        \STATE $\mathcal{F}^0_a, \mathcal{F}^0_b \leftarrow$ fields from sample pairs
        \STATE $\mathcal{L}_{\text{contrast}} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \|\mathcal{F}^0_a(i) - \mathcal{F}^0_b(i)\|_2$
        
        \STATE // Step 6: Causal consistency loss
        \STATE $\mathcal{F}_{\text{env1}}, \mathcal{F}_{\text{env2}} \leftarrow$ fields from different environments
        \STATE $\mathcal{L}_{\text{inv}} = \|\mathcal{F}_{\text{env1}} - \mathcal{F}_{\text{env2}}\|_2^2$
        
        \STATE // Step 7: Counterfactual evolution loss
        \STATE $\text{Mask} \leftarrow$ random intervention mask
        \STATE $\mathcal{F}^{K}_{\text{do}} = \text{Intervene}(\mathcal{F}^t, \text{Mask})$
        \STATE $\mathcal{F}^{K}_{\text{do}} = \text{Propagate}(\mathcal{F}^{t}_{\text{do}}, K)$
        \STATE $\mathcal{L}_{\text{cf}} = \frac{1}{K}\sum_{k=1}^{K} \|\mathcal{F}^{k} - \mathcal{F}^{k}_{\text{do}}\|_2^2$
        
        \STATE // Step 8: Structure regularization
        \STATE $\mathcal{L}_{\text{sparsity}} = |G|_1$
        \STATE $\mathcal{L}_{\text{smoothness}} = |\nabla \mathcal{F}^K|_2^2$
        
        \STATE // Step 9: Total loss
        \STATE $\mathcal{L} = \mathcal{L}_{\text{contrast}} + \lambda_1 \mathcal{L}_{\text{inv}} + \lambda_2 \mathcal{L}_{\text{cf}} + \lambda_3 \mathcal{L}_{\text{sparsity}} + \lambda_4 \mathcal{L}_{\text{smoothness}}$
        
        \STATE // Step 10: Backpropagation
        \STATE $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
    \ENDFOR
\ENDFOR

\RETURN $\theta$
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 2: Causal Field Construction via Gaussian Interpolation

```latex
\begin{algorithm}
\caption{Continuous Causal Field Construction}
\begin{algorithmic}[1]
\REQUIRE Discrete patch features $\{z_i\}_{i=1}^N$, patch positions $\{p_i\}_{i=1}^N$,
          query positions $\{q_j\}_{j=1}^Q$, Gaussian bandwidth $\sigma$
\ENSURE Continuous field values $\mathcal{F}(q_j)$ at all query positions

\FOR{each query position $q_j$}
    \STATE // Compute Gaussian weights
    \FOR{each patch position $p_i$}
        \STATE $d_{ij} = \|q_j - p_i\|_2^2$  \COMMENT{Squared Euclidean distance}
        \STATE $w_{ij} = \exp(-d_{ij} / (2\sigma^2))$
    \ENDFOR
    
    \STATE // Normalize weights
    \STATE $w_{ij} \leftarrow w_{ij} / \sum_{i'=1}^N w_{ij'}$
    
    \STATE // Interpolate field value
    \STATE $\mathcal{F}(q_j) = \sum_{i=1}^N w_{ij} \cdot z_i$
\ENDFOR

\RETURN $\{\mathcal{F}(q_j)\}_{j=1}^Q$
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 3: Directional Influence Function Learning

```latex
\begin{algorithm}
\caption{Directional Influence Function Computation}
\begin{algorithmic}[1]
\REQUIRE Field states $\mathcal{F} \in \mathbb{R}^{H \times W \times C}$,
          projection matrices $W_Q, W_K, W_V$, direction bias $b$,
          sparsity gate parameters $\theta_{\text{sparse}}$
\ENSURE Influence matrix $G \in \mathbb{R}^{N \times N}$, updated field $\mathcal{F}'$

\STATE // Reshape field to patch sequence
\STATE $N \leftarrow H \times W$
\STATE $\mathcal{F}_{\text{flat}} \leftarrow \text{reshape}(\mathcal{F}, [N, C])$

\STATE // Compute multi-head queries, keys, values
\STATE $Q = \text{MultiHead}(\mathcal{F}_{\text{flat}} W_Q, \text{num\_heads})$  \COMMENT{$[H_{\text{head}}, N, D]$}
\STATE $K = \text{MultiHead}(\mathcal{F}_{\text{flat}} W_K, \text{num\_heads})$  \COMMENT{$[H_{\text{head}}, N, D]$}
\STATE $V = \text{MultiHead}(\mathcal{F}_{\text{flat}} W_V, \text{num\_heads})$  \COMMENT{$[H_{\text{head}}, N, D]$}

\STATE // Compute asymmetric influence scores
\FOR{head $h = 1$ to $H_{\text{head}}$}
    \STATE $S^{(h)} = Q^{(h)} (K^{(h)})^T / \sqrt{D}$  \COMMENT{$[N, N]$}
    \STATE $S^{(h)} \leftarrow S^{(h)} + b$  \COMMENT{Add directional bias}
\ENDFOR

\STATE // Apply sparsity gate
\FOR{each pair $(i, j)$}
    \STATE $s_{ij} = \text{Sigmoid}(\text{MLP}_{\theta_{\text{sparse}}}([\mathcal{F}_i, \mathcal{F}_j]))$
    \STATE $G_{ij} \leftarrow \text{Softmax}(S)_{ij} \cdot s_{ij}$
\ENDFOR

\STATE // Aggregate values
\STATE $\mathcal{F}' = G^T V_{\text{flat}}$

\STATE // Residual connection and projection
\STATE $\alpha \leftarrow \text{Sigmoid}(w_{\text{residual}})$
\STATE $\mathcal{F}' \leftarrow \alpha \cdot \mathcal{F}' + (1-\alpha) \cdot \mathcal{F}' W_{\text{out}}$

\RETURN $G, \mathcal{F}'$
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 4: Causal Field Propagation

```latex
\begin{algorithm}
\caption{Causal Effect Propagation on Field}
\begin{algorithmic}[1]
\REQUIRE Initial field $\mathcal{F}^0 \in \mathbb{R}^{H \times W \times C}$,
          number of steps $K$
\ENSURE Field trajectory $\{\mathcal{F}^k\}_{k=0}^K$ after propagation

\STATE $\mathcal{F}^{\text{current}} \leftarrow \mathcal{F}^0$
\STATE $\text{trajectory} \leftarrow [\mathcal{F}^{\text{current}}]$

\FOR{step $k = 1$ to $K$}
    \STATE // Compute influence matrix at current state
    \STATE $G^{(k)}, \mathcal{F}^{(k)}_{\text{infl}} \leftarrow \text{InfluenceFunc}(\mathcal{F}^{\text{current}})$
    
    \STATE // Propagate causal effects
    \STATE $\mathcal{F}^{\text{prop}} = (G^{(k)})^T \cdot \mathcal{F}^{\text{current}}$
    
    \STATE // Update with residual connection
    \STATE $\alpha \leftarrow \text{Sigmoid}(w_{\text{residual}})$
    \STATE $\mathcal{F}^{\text{current}} \leftarrow \alpha \cdot \mathcal{F}^{\text{prop}} + (1-\alpha) \cdot \mathcal{F}^{(k)}_{\text{infl}}$
    
    \STATE $\text{trajectory.append}(\mathcal{F}^{\text{current}})$
\ENDFOR

\RETURN $\text{trajectory}$
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 5: Field-Based Counterfactual Reasoning

```latex
\begin{algorithm}
\caption{Counterfactual Reasoning via Field Intervention}
\begin{algorithmic}[1]
\REQUIRE Initial field $\mathcal{F}^0$, intervention type $\tau$, 
          intervention parameters $\rho$ (position, radius, direction),
          propagation steps $K$
\ENSURE Factual field $\mathcal{F}^{K}$, counterfactual field $\mathcal{F}^{K}_{\text{do}}$

\STATE // Step 1: Factual propagation
\STATE $\{\mathcal{F}^k\}_{k=0}^K \leftarrow \text{Propagate}(\mathcal{F}^0, K)$
\STATE $\mathcal{F}^{K} \leftarrow \mathcal{F}^K$

\STATE // Step 2: Apply intervention at time $t$
\STATE $t \leftarrow \lfloor K/2 \rfloor$  \COMMENT{Intervene at middle step}
\STATE $\mathcal{F}^{t}_{\text{do}} \leftarrow \mathcal{F}^t$

\IF{$\tau = \texttt{remove}$}
    \STATE // Object removal: set to null state
    \STATE $\text{Mask} \leftarrow \text{CreateSpatialMask}(\rho.\text{position}, \rho.\text{radius})$
    \STATE $\mathcal{F}^{t}_{\text{do}} \leftarrow (1-\text{Mask}) \odot \mathcal{F}^{t} + \text{Mask} \odot \mathbf{0}$
\ELIF{$\tau = \texttt{modify}$}
    \STATE // Attribute modification: transform field values
    \STATE $\text{Mask} \leftarrow \text{CreateSpatialMask}(\rho.\text{position}, \rho.\text{radius})$
    \STATE $\mathcal{F}^{t}_{\text{transformed}} \leftarrow \text{MLP}(\mathcal{F}^t)$
    \STATE $\mathcal{F}^{t}_{\text{do}} \leftarrow (1-\text{Mask}) \odot \mathcal{F}^t + \text{Mask} \odot \mathcal{F}^{t}_{\text{transformed}}$
\ENDIF

\STATE // Step 3: Counterfactual rollout
\FOR{step $k = t+1$ to $K$}
    \STATE $G^{(k)}, \mathcal{F}^{(k)}_{\text{infl}} \leftarrow \text{InfluenceFunc}(\mathcal{F}^{k-1}_{\text{do}})$
    \STATE $\mathcal{F}^{k}_{\text{do}} = (G^{(k)})^T \cdot \mathcal{F}^{k-1}_{\text{do}}$
\ENDFOR

\STATE // Step 4: Compare factual vs counterfactual
\STATE $\Delta \mathcal{F} = |\mathcal{F}^{K} - \mathcal{F}^{K}_{\text{do}}|$

\RETURN $\mathcal{F}^{K}, \mathcal{F}^{K}_{\text{do}}, \Delta \mathcal{F}$
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 6: Contrastive Factor Discovery (Implicit)

```latex
\begin{algorithm}
\caption{Implicit Factor Discovery via Contrastive Field Learning}
\begin{algorithmic}[1]
\REQUIRE Sample pairs $\{(x_a^{(i)}, x_b^{(i)})\}_{i=1}^M$ with large semantic differences,
          field construction function $\text{BuildField}(\cdot)$
\ENSURE Spatial significance map $S(p)$ indicating causal factor locations

\FOR{each contrastive pair $(x_a, x_b)$}
    \STATE // Build fields for both samples
    \STATE $\mathcal{F}_a \leftarrow \text{BuildField}(x_a)$  \COMMENT{$[H \times W \times C]$}
    \STATE $\mathcal{F}_b \leftarrow \text{BuildField}(x_b)$  \COMMENT{$[H \times W \times C]$}
    
    \STATE // Compute spatial difference
    \STATE $\Delta_{ab}(p) = \|\mathcal{F}_a(p) - \mathcal{F}_b(p)\|_2$  \COMMENT{$\forall p \in \text{spatial grid}$}
    
    \STATE // Accumulate differences
    \STATE $S(p) \leftarrow S(p) + \Delta_{ab}(p)$
\ENDFOR

\STATE // Normalize significance map
\STATE $S(p) \leftarrow S(p) / \max_{p'} S(p')$

\RETURN $S(p)$  \COMMENT{High values indicate causal factor locations}
\end{algorithmic}
\end{algorithm}
```

---

## Algorithm 7: Causal Graph Extraction from Influence Matrix

```latex
\begin{algorithm}
\caption{Discrete Causal Graph Extraction}
\begin{algorithmic}[1]
\REQUIRE Influence matrix $G \in \mathbb{R}^{N \times N}$, threshold $\tau$
\ENSURE Adjacency matrix $A \in \{0, 1\}^{N \times N}$

\FOR{$i = 1$ to $N$}
    \FOR{$j = 1$ to $N$}
        \IF{$G_{ij} > \tau$ AND $i \neq j$}
            \STATE $A_{ij} \leftarrow 1$  \COMMENT{Edge from $i$ to $j$}
        \ELSE
            \STATE $A_{ij} \leftarrow 0$
        \ENDIF
    \ENDFOR
\ENDFOR

\RETURN $A$
\end{algorithmic}
\end{algorithm}
```

---

## Complexity Analysis

### Time Complexity

1. **Field Construction** (Algorithm 2):
   - For $N$ patches and $Q$ query positions: $O(N \cdot Q)$
   - With grid queries ($Q = N$): $O(N^2)$

2. **Influence Function** (Algorithm 3):
   - Multi-head attention: $O(N^2 \cdot C)$
   - Sparsity gate: $O(N^2 \cdot C)$
   - Total: $O(N^2 \cdot C)$

3. **Propagation** (Algorithm 4):
   - For $K$ steps: $O(K \cdot N^2 \cdot C)$

4. **Counterfactual** (Algorithm 5):
   - Factual + counterfactual propagation: $O(2K \cdot N^2 \cdot C) = O(K \cdot N^2 \cdot C)$

### Space Complexity

1. **Field Storage**: $O(N \cdot C)$
2. **Influence Matrix**: $O(N^2)$
3. **Propagation Trajectory**: $O(K \cdot N \cdot C)$

### Comparison with MLLM-CD

| Aspect | MLLM-CD | MetaCausalField |
|--------|----------|-----------------|
| Pipeline | 3-stage (serial) | End-to-end (unified) |
| Factor Discovery | Explicit discrete variables | Implicit continuous field |
| Structure Learning | Static graph | Dynamic propagation |
| Counterfactuals | Structure refinement only | Direct trajectory learning |
| Complexity | $O(3 \cdot N^2)$ | $O(K \cdot N^2 \cdot C)$ |
| Memory | Multiple stages | Single pass |

---

## Implementation Notes

1. **Efficient Computation**: The $O(N^2)$ complexity in influence computation can be optimized using:
   - Sparse attention mechanisms
   - Hierarchical aggregation
   - Spatial locality constraints

2. **Memory Management**: For large $N$, implement:
   - Gradient checkpointing during propagation
   - Chunked computation of influence matrix
   - Mixed-precision training

3. **Parallelization**:
   - Batch processing across samples
   - Multi-head attention parallelization
   - GPU-accelerated matrix operations

4. **Numerical Stability**:
   - Use LayerNorm in influence function
   - Clip gradients during backpropagation
   - Add epsilon to normalization operations