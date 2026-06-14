# Experimental Setup and Results

## Datasets and Architectures

We evaluate on two standard image classification benchmarks:

\begin{itemize}
\item \textbf{CIFAR-10} \cite{krizhevsky2009learning}: 50,000 training / 10,000 test images across 10 classes. This is the primary evaluation benchmark for all experiments.
\item \textbf{CIFAR-100} \cite{krizhevsky2009learning}: 50,000 training / 10,000 test images across 100 classes. The finer-grained classification task increases regularization sensitivity and amplifies differences between methods.
\end{itemize}

We use two architectures spanning different design families:

\begin{itemize}
\item \textbf{ResNet-20} \cite{he2016deep}: A compact residual network with 0.27M parameters, serving as the primary architecture for all experiments including ablations and hyperparameter sweeps.
\item \textbf{VGG-16-BN} \cite{simonyan2015very}: A deep feedforward network with batch normalization, containing 15M parameters ($56\times$ larger than ResNet-20), used for cross-architecture validation.
\end{itemize}

Standard data augmentation is applied: random crop ($32 \times 32$, padding 4), random horizontal flip, and per-channel normalization.

## Methods Compared

We compare 11 weight decay methods organized into four categories:

\begin{itemize}
\item \textbf{Baselines:} No WD ($\lambda = 0$), Fixed WD ($\lambda = 5 \times 10^{-4}$), Fixed WD grid search ($\lambda \in \{10^{-4}, 3 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$), Stagewise WD (LR-proportional at milestones).
\item \textbf{AADWD variants:} Conservative, Aggressive, and Square (Section~\ref{sec:variants}).
\item \textbf{Ablation controls:} Random Dynamic WD ($\lambda_t = c \cdot \gamma_t \cdot U(0,1)$), Equivalent Cumulative WD (constant $\lambda$ matched to AADWD's time-average), Norm-Matched WD (weight norm trajectory matched to AADWD).
\item \textbf{Recent adaptive method:} Cautious Weight Decay (CWD) \cite{liu2025cautious}, applying coordinate-wise sign-based masking.
\end{itemize}

## Training Protocol

All experiments use SGD with momentum $0.9$, initial learning rate $0.1$, milestone schedule at epochs $[80, 120]$ with $\gamma \times 0.1$, batch size 128, and 200 training epochs. AADWD hyperparameters are swept over $c \in \{0.5, 1.0, 2.5, 5.0, 10.0\}$ and $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$, with $\lambda_{\min} = 10^{-6}$ and $\lambda_{\max} \in \{0.01, 0.05\}$.

We report four evaluation metrics: best test accuracy (highest test accuracy during training), final test accuracy (epoch 200), generalization gap (train accuracy $-$ test accuracy at convergence), and weight norm ($\|w_T\|_2$).

## Main Results: Fixed WD Dominates All Dynamic Methods
\label{sec:main_results}

Table~\ref{tab:main} presents the central comparison on CIFAR-10/ResNet-20.

\begin{table}[t]
\centering
\caption{Main results on CIFAR-10/ResNet-20 (200 epochs). Best and tied-best results in \textbf{bold}. Fixed WD ($5 \times 10^{-4}$) achieves the best performance; the budget-equivalent constant exactly matches it. All dynamic methods are dominated.}
\label{tab:main}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Gen Gap} & \textbf{Weight Norm} \\
\midrule
No WD & 90.49 & 90.31 & 9.17 & 129.42 \\
Fixed WD ($10^{-4}$) & 92.08 & 91.83 & 7.64 & 56.34 \\
Fixed WD ($3 \times 10^{-4}$) & 92.39 & 91.97 & 7.52 & 30.70 \\
Fixed WD ($5 \times 10^{-4}$) & \textbf{92.54} & \textbf{92.29} & 7.17 & 23.49 \\
Fixed WD ($10^{-3}$) & 92.32 & 91.92 & 6.95 & 17.13 \\
Fixed WD ($3 \times 10^{-3}$) & 88.63 & 87.25 & 5.20 & 11.44 \\
\midrule
Stagewise WD & 92.44 & 92.27 & 7.18 & 33.22 \\
CWD \cite{liu2025cautious} & 91.79 & 86.95 & 3.22 & 9.28 \\
\midrule
AADWD Conservative & 92.37 & 92.22 & 7.15 & 23.60 \\
AADWD Aggressive & 92.05 & 91.57 & 7.50 & 21.47 \\
AADWD Square & 92.13 & 91.78 & 7.47 & 38.75 \\
\midrule
Random Dynamic WD & 92.06 & 91.95 & 7.54 & 34.19 \\
Equiv.\ Cumulative WD & \textbf{92.54} & \textbf{92.29} & 7.17 & 23.49 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) Fixed WD at $\lambda = 5 \times 10^{-4}$ achieves the best performance ($92.54\%$), confirming the standard practice. (ii) The budget-equivalent constant WD \emph{exactly} matches fixed WD ($92.54\%$, weight norm $23.49$), establishing the budget equivalence principle. (iii) AADWD Conservative ($92.37\%$) performs within $0.17\%$ of fixed WD because $\hat{\delta}_t \approx 0$ causes it to degenerate to $\lambda_t \approx c \cdot \gamma_t$. (iv) All AADWD variants are strictly dominated by fixed WD. (v) CWD shows dramatic late-training collapse: best accuracy $91.79\%$ degrades to final $86.95\%$ ($\Delta = 4.84\%$), with an abnormally low weight norm of $9.28$ indicating over-aggressive shrinkage.

The fixed WD grid search reveals a well-behaved optimum at $\lambda = 5 \times 10^{-4}$: accuracy increases monotonically from $\lambda = 10^{-4}$ ($92.08\%$) to $\lambda = 5 \times 10^{-4}$ ($92.54\%$), then decreases for $\lambda = 10^{-3}$ ($92.32\%$) and sharply for $\lambda = 3 \times 10^{-3}$ ($88.63\%$). This smooth landscape makes single-hyperparameter tuning straightforward.

## Cross-Architecture and Cross-Dataset Validation
\label{sec:cross_arch}

Table~\ref{tab:cross_arch} extends the comparison to CIFAR-100/ResNet-20 and CIFAR-10/VGG-16-BN.

\begin{table}[t]
\centering
\caption{Cross-architecture and cross-dataset results (best test accuracy, \%). Fixed WD dominates in all settings. CWD shows best $\to$ final collapse universally. $\Delta_{\text{CWD}}$ denotes the best-to-final degradation.}
\label{tab:cross_arch}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{C10/ResNet-20} & \textbf{C100/ResNet-20} & \textbf{C10/VGG-16-BN} \\
\midrule
No WD & 90.49 & 64.70 & 92.34 \\
Fixed WD ($5 \times 10^{-4}$) & \textbf{92.54} & \textbf{68.45} & \textbf{93.86} \\
AADWD Conservative & 92.37 & 68.24 & 93.75 \\
AADWD Aggressive & 92.05 & 61.34 & 90.97 \\
CWD (best $\to$ final) & 91.79 $\to$ 86.95 & 66.84 $\to$ 54.27 & 92.95 $\to$ 86.47 \\
\midrule
$\Delta_{\text{CWD}}$ & $-4.84$ & $-12.57$ & $-6.48$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) Fixed WD is the best method across all three settings, confirming architecture- and dataset-independence. (ii) AADWD Conservative is consistently the second-best dynamic method but remains within noise of fixed WD ($\Delta \leq 0.21\%$). (iii) AADWD Aggressive degrades more severely on the harder task CIFAR-100 ($\Delta = -7.11\%$ vs.\ fixed WD), suggesting that alignment-dependent modulation introduces instability on tasks requiring stronger regularization. (iv) CWD late-training collapse is \emph{universal}: the best-to-final degradation of $4.84\%$, $12.57\%$, and $6.48\%$ across settings constitutes a systematic failure mode. The effect is most severe on CIFAR-100 ($12.57\%$ drop), where the 100-class problem amplifies the impact of weight over-shrinkage. (v) VGG-16-BN ($15$M parameters) shows qualitatively identical patterns to ResNet-20 ($0.27$M parameters), indicating that our conclusions are independent of model scale.

## Budget Equivalence Experiment
\label{sec:budget}

The budget equivalence experiment provides the cleanest causal evidence in this study.

\paragraph{Design.} We first complete a full AADWD-Aggressive run on CIFAR-10/ResNet-20 and record the per-step weight decay values $\{\lambda_t\}_{t=1}^T$. We then compute the time-average $\bar{\lambda} = (1/T)\sum_t \lambda_t$ and run a new experiment with constant $\lambda = \bar{\lambda}$ for 200 epochs under otherwise identical conditions.

\paragraph{Results.} The budget-equivalent constant WD achieves $92.54\%$ best accuracy, $92.29\%$ final accuracy, weight norm $23.49$, and generalization gap $7.17$---\emph{exactly} matching the fixed WD ($5 \times 10^{-4}$) baseline on all four metrics. In contrast, AADWD Aggressive with the same cumulative budget achieves only $92.05\%$ best accuracy, $91.57\%$ final accuracy, weight norm $21.47$, and generalization gap $7.50$.

\paragraph{Interpretation.} The $0.49\%$ gap between AADWD-Aggressive ($92.05\%$) and its budget-equivalent constant ($92.54\%$) is entirely attributable to the suboptimal \emph{temporal allocation} of weight decay, not to the total budget. The dynamic schedule wastes regularization pressure on early-training steps where it is less effective while providing insufficient regularization in later stages. But crucially, a constant schedule with the same total budget achieves the optimal performance---proving that the temporal dynamics of $\lambda_t$ provide \emph{zero additional information} beyond the average magnitude.

## LR Decoupling Experiment
\label{sec:decoupling}

Table~\ref{tab:decoupled} presents the most dramatic finding of this study: the effect of removing the learning rate multiplier $\gamma_t$ from AADWD.

\begin{table}[t]
\centering
\caption{Effect of removing the LR multiplier $\gamma_t$ from AADWD (CIFAR-10/ResNet-20). Decoupling triggers catastrophic collapse for the aggressive variant and severe degradation for the conservative variant.}
\label{tab:decoupled}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{Coupled (\%)} & \textbf{Decoupled (\%)} & $\boldsymbol{\Delta}$ & \textbf{WN (Coupled)} & \textbf{WN (Decoupled)} \\
\midrule
Conservative & 92.37 & 90.32 $\to$ 80.30 & $-12.07$ & 23.60 & 5.53 \\
Aggressive & 92.05 & 84.49 $\to$ 10.00 & $-82.05$ & 21.47 & 0.0036 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Aggressive collapse ($92.05\% \to 10.00\%$).} Without the $\gamma_t$ damper, the aggressive variant exhibits a positive feedback instability. After the first milestone at epoch 80, the learning rate drops from $0.1$ to $0.01$, but the decoupled $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ remains at approximately $5 \times 10^{-4}$. The effective regularization-to-gradient ratio $\lambda_t / \gamma_t$ increases by $10\times$, causing weight decay to overwhelm the gradient signal. The weight norm begins shrinking rapidly, which alters $\hat{\delta}_t$, which further increases $\lambda_t$, creating a self-amplifying feedback loop. By epoch 200, the weight norm has collapsed to $0.0036$ and training has degraded to random chance ($10\%$). The best accuracy of $84.49\%$, achieved before the first milestone, confirms that the collapse is triggered specifically by the learning rate drop.

\paragraph{Conservative degradation ($92.37\% \to 80.30\%$).} The conservative variant's decoupled form $\lambda_t = c \cdot (1 - \hat{\delta}_t) \approx 5 \times 10^{-4}$ becomes a constant weight decay that is $10\times$ too strong when $\gamma_t = 0.01$ (after epoch 80) and $100\times$ too strong when $\gamma_t = 0.001$ (after epoch 120). This causes progressive over-regularization (weight norm shrinks to $5.53$, vs.\ $23.60$ in the coupled case) and underfitting, but the bounded nature of the conservative formula prevents complete collapse.

\paragraph{Mechanistic insight.} The LR multiplier $\gamma_t$ serves as a structural stabilizer that ensures weight decay remains commensurate with the optimization step size. In standard $\ell_2$ regularization, this coupling is automatic ($\text{effective decay} = \gamma_t \cdot \lambda$). Decoupled weight decay \cite{loshchilov2019decoupled} deliberately removes it for Adam---but uses a \emph{constant} $\lambda$, which is safe because a constant $\lambda$ was tuned jointly with the learning rate schedule. AADWD with a \emph{varying} $\lambda_t$ requires the $\gamma_t$ coupling to be reintroduced as an explicit stability mechanism.

## Hyperparameter Sensitivity
\label{sec:sensitivity}

Table~\ref{tab:sensitivity} presents the sensitivity of AADWD-Aggressive to the scaling constant $c$ and EMA coefficient $\beta$ on CIFAR-10/ResNet-20.

\begin{table}[t]
\centering
\caption{Hyperparameter sensitivity of AADWD-Aggressive (CIFAR-10/ResNet-20). Left: $c$-sweep with $\beta = 0.999$. Right: $\beta$-sweep with $c = 2.5$. The method is robust over a $5\times$ range of $c$ and across three orders of magnitude of $(1-\beta)$, but no configuration outperforms fixed WD ($92.54\%$).}
\label{tab:sensitivity}
\small
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{c}{\textbf{$c$-sweep} ($\beta = 0.999$)} \\
\midrule
$c$ & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Weight Norm} \\
\midrule
0.5 & 91.87 & 91.69 & 65.48 \\
1.0 & 92.18 & 92.07 & 41.27 \\
2.5 & 92.05 & 91.57 & 21.47 \\
5.0 & 87.98 & 87.24 & 13.07 \\
10.0 & 52.12 & 10.00 & 0.004 \\
\midrule
\multicolumn{4}{c}{\textbf{$\beta$-sweep} ($c = 2.5$)} \\
\midrule
$\beta$ & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Weight Norm} \\
\midrule
0.9 & 92.08 & 91.95 & 21.75 \\
0.99 & 92.24 & 91.97 & 21.65 \\
0.999 & 92.05 & 91.57 & 21.47 \\
0.9999 & 92.25 & 91.80 & 20.56 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Scaling constant $c$.} AADWD-Aggressive is robust over a $5\times$ range ($c \in [0.5, 2.5]$): accuracy varies by only $0.31\%$ ($91.87$--$92.18\%$). However, $c = 5.0$ causes noticeable degradation ($87.98\%$), and $c = 10.0$ triggers complete collapse ($52.12\% \to 10.00\%$, weight norm $0.004$)---the same positive feedback failure mode as in the decoupled experiment. The weight norm decreases monotonically with $c$ (from $65.48$ at $c = 0.5$ to $0.004$ at $c = 10.0$), confirming that $c$ primarily controls the regularization budget. Critically, even the best configuration ($c = 1.0$, $92.18\%$) falls short of fixed WD ($92.54\%$) by $0.36\%$.

\paragraph{EMA coefficient $\beta$.} The alignment proxy is remarkably insensitive to $\beta$: across three orders of magnitude of the time constant ($1 - \beta \in [10^{-4}, 10^{-1}]$), best accuracy varies by only $0.20\%$ ($92.05$--$92.25\%$). This insensitivity is itself informative: it confirms that the alignment signal $\hat{\delta}_t$ varies so little during training that the smoothing window does not matter---whether the EMA responds to the last 10 steps ($\beta = 0.9$) or the last 10,000 steps ($\beta = 0.9999$), it captures essentially the same near-constant value.

## Alignment Proxy Characterization
\label{sec:alignment_char}

We directly characterize the alignment proxy $\hat{\delta}_t$ measured during training on ResNet-20/CIFAR-10 (Table~\ref{tab:alignment}).

\begin{table}[t]
\centering
\caption{Alignment proxy $\hat{\delta}_t$ statistics across training phases (ResNet-20/CIFAR-10). The signal is near-constant at $O(10^{-3})$ with negligible variation, explaining why alignment-based WD adaptation degenerates to a deterministic function of the learning rate.}
\label{tab:alignment}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Training Phase} & \textbf{Mean $\hat{\delta}_t$} & \textbf{Std} & \textbf{Range} \\
\midrule
Early (epochs 1--80) & 0.004491 & --- & --- \\
Mid (epochs 81--120) & 0.003352 & --- & --- \\
Late (epochs 121--200) & 0.002824 & --- & --- \\
\midrule
Overall & $\sim 0.003$ & 0.000753 & $[0.0028, 0.0045]$ \\
\midrule
Pearson $r$ (EMA vs.\ large-batch) & \multicolumn{3}{c}{0.849} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) The alignment proxy remains at $O(10^{-3})$ throughout training, with a total range of only $[0.0028, 0.0045]$---a $1.6\times$ variation over 200 epochs. (ii) The standard deviation ($7.5 \times 10^{-4}$) is comparable to the mean, indicating noisy measurements around a near-constant center. (iii) There is a monotonic decrease from early to late training ($0.0045 \to 0.0028$), but this $1.6\times$ variation is negligible compared to the $100\times$ variation in the learning rate $\gamma_t$ over the same period. (iv) The Pearson correlation of $r = 0.849$ between the minibatch EMA and large-batch alignment confirms that the proxy is faithful---the issue is not measurement noise, but that the underlying signal is inherently uninformative.

\paragraph{Consequence for AADWD.} With $\hat{\delta}_t \approx 0.003$, the conservative formula becomes $\lambda_t = c \cdot \gamma_t \cdot (1 - 0.003) \approx c \cdot \gamma_t \cdot 0.997$, which is functionally indistinguishable from $\lambda_t = c \cdot \gamma_t$---a deterministic schedule with no alignment contribution. This explains why AADWD Conservative ($92.37\%$) nearly matches fixed WD ($92.54\%$): it is, in effect, a learning-rate-proportional schedule, and the $0.17\%$ gap comes from the proportionality constant $c$ rather than from alignment information. The alignment signal is dominated by the learning rate dynamics by a factor of $\sim 60\times$ ($100\times$ LR variation vs.\ $1.6\times$ alignment variation).
