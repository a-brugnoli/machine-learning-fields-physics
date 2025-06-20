# For basic learining techniques to various theories of information representations in feilds physics

Wheather forecasting: 
- Graphcast: https://arxiv.org/abs/2012.12794
- ClimaX: https://arxiv.org/abs/2301.10343
- Pangu-Weather 
- FourcastNet
- Neural General Circulation Model (NGCM): https://arxiv.org/abs/2311.07222
- Aurora: https://arxiv.org/abs/2405.13063

Drug Discovery:
- Alphafold
- Drug Design

Reasoning engines
- High school level problems

## ODE and neural nets
Resnet and Euler's method:

$$
\min L(x, \theta), \qquad \text{with constraint } \quad
x_{t+1} = x_t + f(x_t, \theta_t).$$
Forward computation of the graph of ResNet

Disctize and then optimize. 

Chen et al. popularize the optimize then discretize approach in 2018.

## NN for PDEs

### Discrete Neural models

#### Learning from partial observations: regular grids

$$
\begin{aligned}
\dot{x} = & f(x, t) \qquad x(t_0) =x_0, \\
y = & h(x)
\end{aligned}
$$

Learning problem: least squares problem from observations

Evolution function is parameterized by a neural network $f_\theta(x, t)$.

Better than working directly in the observation space.

#### Irregular grids

Mesh nodes are mapped to a graph which is processed by a GNN.
The GNN acts as a time stepper neural ODE solver.

Message passing PDE soler
Objectieforecast spatio-temporal dynamics

Framework Encode-process-decode
Map data from physical space to latent space, process in latent space, and decode back to physical space.

Encode: Take as input last $K$ data values at each node $i$
$$
f_i^0 = \mathrm{Encode}(u_i^{k-K}, \dots, u_i^k)
$$

Process: in latent space $M$ message passing steps $f_i^m$

Decode: output last $K$ values

#### Discrete space models: transformers

Leverage recent progrees in NLP and computer vision.

Modelling evvolution of a video is close to modelling a spatio-temporal process.

Transformers operate on a tokenized representation of the data: either in the physical space or most-often in the latent space.

Two core attention mechanisms:
- Self-attention: linear transformation. Captures contextual representation of the inputs. Complexity $N^2$ where $N$ is the size of the input.
- Cross-attention: compute attention between tokens in the sequence and a set of query tokens.

Vision transformers:
ViT: split image into patches. Embeds them linearly (flatten) and feed them into a transformer.
- Stacked encoders could be used for time stepping

Simple autoregressive transformer (decode in GTP/Llama)


BCAT model (Liu et al. 2025):
- Patchify and flatten the input image
- MLP encoder
- BLock causal transformer
- MLP decoder

Regular patches do not capture the physics.

Transformer Transolver (Wu et al. 2024)
- Physical tokens: decompose automatically the mesh into domains where points share similar physical states
- Decrease attention complexity. Apply attention on the physical tokens instead of regular patches.


#### Continuous space models
Neuro-Operators: instead of learning maps between vector spaces learn maps between function spaces.

Objectives: 
- learn irregular and diverse geometries;
- query at any space-time coordinate in the output space;

Three families of methods:
- Fourier neural operators (FNO): learn a Fourier basis of the function space.
- Implicit neural operators (INO): learn a basis of the function space using a neural network. Coral (2023)
Attention mechanics + transformer architecture: AROMA (2024)


Fourier neural operator (Li et al. 2021)
- Function spaces $\mathcal{V}(\Omega; \mathbb{R}^n)$ and $\mathcal{U}(\widehat{\Omega}; \mathbb{R}^m)$ (the domains $\Omega$ and $\widehat{\Omega}$ are not necessarily the same).
- ${G}: \mathcal{U} \rightarrow \mathcal{V}$ is a nonlinear operator.

Classical Neural Network $u, v$ are vectors.

Neural operator: $v, u$ are functions, not vectors.
$$
v_{t+1}(x) = K_t(u(x))
$$


CORAL: coordinate based operator learning

AROMA: active reduced order model with attention
Tokens encode local spatial information. Cross attention between $T_{\rm geo}$ and $x$

## Takeaways
- Scaling is a central problem
- Latent models are one promising way to proceed
- Importance of efficient and realiable "physical" encodings-decodings operating on multiple geometries


# Linear and nonlinear system identification (Loiseau)

In control:
- what is physically realizable?
- what are the operating conditions?
- which mechanisms to use? Sensors, actuators, etc.
- how do I handle measurement noise? Disturbances? What if an actuator fails?
- what are the actuactors specs?
- model based or data driven? How accurate do I need to be? Can I quantify the uncertainty? How expensive is to run?


## Linear system identification
LTI discrete time system
$$
\begin{aligned}
x_{t+1} = & Ax_t + Bu_t, \\
y_t = & Cx_t + Du_t
\end{aligned}
$$
Linear residual network

Kalman decomposition: divide into controllable and uncontrollable subspaces.

Only the realization of the 
observable and controllable subspaces can be obtained from the data.

The realization of the jointly controllable and observable subspace is not unique (coordinate transformation).

Taking subsequent states makes appear the control Hankel matrix and the controllability matrix. Same for the observability part.

Extended state space model: use the pseudo inverse to express past-output in terms of the state.

$H$ is a toepliz matrix.

$\mathcal{O}A\mathcal{O}^\dag$ is the Koopman operator.

We want to identify the convolution operator, the Hankel operator, and the Koopman operator via least squares of the frobenius norm.

The problem is non-convex. If I use a sufficient number of time steps and the system is stable then one part goes to zero. I can then use an oblique projection to obtain a quadratic program with quadratic constraints.

$M$ is sym. pos. def. and it's a free parameter.

Spectral decomposition of the covariance matrix. Minimize the trace of the covariance matrix leads to the minimum eigenvalue. So we obtain a generalized eigenvalue problem.

Once $\mathcal{O}, H, \Delta$ are known I can obtain $A,B,C,D$

If $M$ is the identify we mimimize the predicition error. One can change $M$ to minimize the mutual information.

The input needs to be rich to excite all frequencies; gaussian white noise.

Example on spring mass damper. This technique can be use on noisy data and can be used to estimate the Kalman filter.

Model predictive control. The problem is convex if $U, Y$ are convex.


### Behavioural approach
Design the control directly from the data.
A system is a tuple
$\Sigma = (\mathbb{Z}_+, \mathbb{W}, \mathcal{B})$

- $\mathbb{Z}_+$ is the discrete time axis
- $\mathbb{W}$ is a signal space
- $\mathcal{B}$ is the behaviour, set of the set of solutions.

Then the system is linear if $\mathbb{W}$ is a vector space and $\mathcal{B} \subset \mathbb{W}^{\mathbb{Z}_+}$ is a subspace

$\mathcal{B} \equiv \{w \in \mathbb{W}^{\mathbb{Z}_+}| R(\sigma)w = 0\}$

where $\sigma$ is a shift operator.


Foundamental lemma of the behavioural approach:
Consider $T, \tau \in \mathbb{Z}_+$ and
- a controllable LTI system
- a $T$ sample long trajectory $\mathrm{col}(u_d, y_d) \in \mathcal{B}_T$ where $u$ is persistent exciting of order $T+N$ ($N$ is the size of the system)
then it holds
$$
\mathrm{colspan}\left( H \begin{pmatrix}u_d \\ y_d \end{pmatrix}\right) \in \mathcal{B}
$$

If I have an LTI model then I can uniquely contruct the Hankel matrix. If I have the Hankel matrix I can obtain an LTI system. The equivalence is not true for generak systems (LTV, LPV)

Then we solve a least square solution and to a batch procedure to predict future values of the output.


This can be used for MPC. If the system is a deterministic LTI then linear MPC and the purely data 

- Parameterizaiton of the input output behaviour is irrelevant as long as it captures the behaviour.
- proper choice of sensors and actuators are important but this a combinatorial problem.

## Nonlinear system identification

Using suitable coordintes many systems are sparse. 
Tlolemaic and Copernicus revolution: putting the sun in the center makes the laws pretty simple.

SINDy paradigm: standard least square will lead to dense dynamics. $L^1$ regularization to obtain sparse dynamics.

The problem becomes a mininization of the $L^1$ norm of the coefficient

Driven cavity problem $Re= 7500$. The flow is quasiperiodic: this means that two oscillators are required. Galerkin based ROM based on projection does not work.

In PDE identification one can discard terms that do not respect the symmetry

Sindy for RL: RL is really data hungry.

Use sindy to obtain a reduced order model the environment.
For the agent a NN is used for the policy. The NN is parametrized by a sparse regression. The sparse regression is also used for the reward function.

Sindy is not a silver bullet. The librairy function requires physical knowledge. Polynomials work well. The constraint require lot of expertise.

# Deep learning for scientists and engineers

The idea is to be able to incorporate seemlessly data and solve ill posed problems. I may not know the parameters and/or bcs.

Crack in a panel. I have only ultrasould video. 

NN don't use polynomials.

Discriminatory function: provides a sign for eveey measure.

The basis depends on the data.

Data need to preprocessed: intput and output need to be nondimensionalized and normalized. Same for the equations.

JAX is the framework for high order derivatives.

Boundary value problem

$$
-\epsilon \frac{d^2 u}{dx^2}+ u = f, \qquad \epsilon \ll 1
$$
To get the boundary layes we set a min max problem. Maximize the weigths that represents the stiffness part.

The ideas is to move to chaotic pattern of the residual (noisy). This leads to zero gradients. In FE the error is the smallest when it is the same in all elements.

Where to find the encoder-decoder limit. After plane $T$ I want to forget my input.

Phase transition in the behavious of the norm of the gradient. We use a Lagrangian with $\lambda$ that indicates the fluctuation.

Weighted residuals: use test space and multiply residual and integrate.. Quadrature or Monte Carlo can be used to compute integrals.

Regularity can be decreased by doing integration by parts.

Classical method have a only approximation error.

Seprable PINN use separation of variables with tensor product.

Kolmogorov superposition theorem: the inner functions are fractals.
In KAN (Kolmogorov-Arnold neural networks) the unknowns are functions.

Inner function can be approximate with MLP.
Replace splines with Chebicev polynomials (or Legendre or wavelets).

KANs are pretty forgiving. They do not submit to noisy data.

$h$ refinement is related to the number of layers in a network (good for rough problems). $p$ refinement is the number of neurons (smooth problems).

Activitation functions are paramemtrize to selectively fire.

How to sample points. Residual adaptive refinement. Just like a posteriori refinement in finite element. Importance sampling.

# Practise

Data structure in DL
- Batch size : number of samples (data size)
- Feature size; number of features (input size)

Let's say Batch is N and features is D. Then we have a matrix of size $N \times D$. In PINN D is the number of independent variables of the PDE.

AD and dimensionality. If input space is $\mathbb{R}^n$ and output space $\mathbb{R}^m$ then

- if $n\gg m$ go for reverse AD
- if $m\gg n$ go for forward AD


# Neural Operators

Functional can be representedd by a single shallow neural network (Approx. of continuous functionals ... Chen and Chen 1993).

Similar story for nonlinear operators (Universal approx. to nonlinear operators ... Chen and Chen 1995): the output of a shallow neural network is muliplied with another shallow neural network (two networks works together in sync).

Trunk is associated to how the operator is expressed in terms of a basis.  

When using deep neural networks the curse of dimensionality is avoided (input space needs to be compact and operator continuous): proof in Karniadakis Nature Machine Intelligence. The assumptions can be relaxed to discontinuous operatos with bounded jumps (Mishra and Karniadakis, 2022).

Feature expansions can account for discontinuity (like in enriched finite elements).

Neural operators can be used to compute Laplace transforms, solve optimal control problems and discover equations. 

Stocastic ODEs like 
$$dy(t; \omega) = k(t; \omega)y(t; \omega) dt$$
wtih the Gaussian process $k(t; \omega) \sim \mathcal{GP}(0, \sigma^2 \exp(-||t_1 -t_2||^2/2l^2))$,
 can be converted into deterministic (Karhunen-Loève expansions).

Given a network one can use symbolic regression to find an analytical expression (PySR).

Classical solver can be mixed with Neural operators especially for very fast dynamics (chemistry for instance).

Trunk provides a almost dependent basis. 
- train trunk first and then orthogonalize with QR
- then train the branch.

A linear operator forced with delta has Green functions as solutions. The solution of a generic problem is then given by the integral equations with Green functions kernel. This motivates an architecture for convolutions.

One can use KAN for the Trunk in order to have an interpretable basis.
Application Temperature to Thermal Stresses (DeepONet + KAN has error 2%).

Neural operator can do transfer learning (different geometries and equations). If singularity arises one needs to do something by eniriching the loss.

# Practise Neural Operator

$$
u' = v, \qquad u|_{\partial\Omega}=0
$$
where $v$ is a Guassian process.

Normalization of data, two ways
- $x_n = (x-u(x))/\sigma(x)$, data driven approach
- $x_n = 2(x-x_{\rm min})/(x_{\rm max} - x_{\rm min})-1$

We are goins to lear $\mathcal{G}(u)(v)$.

Branch net: feed the $v$ sampled at many points (equidistant or not) with data structure $N_{\rm batch} \times N_{\rm features}$. $N_{\rm features}$ correspnds to $v(x_i)$ where $i=1, \ldots, N_{\rm features}$

Trunk net: feed the domain at many points. Since the problem is 1D this is just a row vector of size $N_{\rm sample} \times 1$. $N_{\rm sample}$ corresponds to the number of points in the domain $x_i, i, = 1, \ldots, N_{\rm sample}$. Notice that $i$ and $j$ are not the same.

Output of branch net $N_{\rm batch} \times l_d$ where $l_d$ is the output of the network.
Output of trunk net $N_{\rm sample} \times l_d$ where $l_d$. The we do the contraction on axis 1. 

$l_d$ depends on the number of sample. SVD can be used to find the most energetic modes.


So $u$ will be be of size $N_{\rm batch} \times N_{\rm sample}$.

Einsten summation capabilities.

When doing PINN we dont need to impose phyics on all samples but only on somes

## PIKAN
Chebichev polynomials
$$
T_n(x) = \cos(n\arccos(x))
$$
See the compact code in ChebyKANLayer.

# JAX
Jit is a just-in-time compiler. It sppeeds up the code massively by compiling it to machine code.

Slicing and if logic cause problem with jit. But there are workarounds

New data structure pytree.

Check Equinox for many more features (NN like pytorch and many more).

# Generalization of learning for physics

Implicit models: capture correlation between data and not between explicit physical variables.

Physical models; capture causality and are developed from first principles.

Physics complexity is orders of magnitude higher than in classical ML tasks.

Nature does not shuffle data. We do. Shuffling data entails a loss of information.

0 shot generalization;
- several environments;
- invariant representation;
- 0 shot means do not use new data.

Approches for 0 shot:
- robust estimation;
- invariant risk minimization;
- risk extrapolation;

Few shot generalization:
- learn from several environments;
- learn environments conditionned problems;
- learn with scarse data;

Approaches for few shot:
- meta learning;
- learn shared parameters from several environments;
- double optimization process.

## Solving parameteric PDEs

Data from simulations of PDEs.
An Environment is: physics (muliple PDEs), initial condition, boundary conditions, parameters.

Training: sample environment parameters and trajectories from each enviroment.

### Conditional adaptation

Assumption:
- All domains share the same form of dynamics (same PDE). One domain corresponds to a specific instance of the PDE.

Two exammples:
- meta-learning: Generalizing to new Physics via Context informed dynamics. (NeurIPS 2024)
- Adaptive conditioning.

CODA-framework

How to: 
- learn to condition the learned function to the environment;
- in this way it can adapt fast with a few samples to a new environment.
- Adaptation rule
$$
\theta^e = \theta^c + \delta \theta^e
$$
$\theta^c$ shared parameters across environments.

Training objective: 

$$
\min_{\theta^c, \delta\theta^e} \sum_{e \in D_{\rm train}} ||\delta \theta^e||
$$
Local constraint: $\delta\theta^e$ is in the neighborhood of $\theta^c$.

Small rank adaptation: $\delta\theta^e$ has small intrinsic dimensionality

$\delta\theta^e$ is generated by a hypernetwork.

ZEBRA: in context pretraining (Serrano et al. 2025).
Context trajectory: trajectory from same PDE but different initial condtion.
No optimization performed for the pretraining.

### Generative approaches

ENMA: tokenwise autoregression for generative Neural PDE operators (Kassai et al. 2025)
This paradigm explores autoregressive models in continuous space (most suitable for physics).

Idea of flow matching to learn a probability path in time. This is useful to predict next pixel in a figure.

# Statical tour of physical informed ML (Claire Boyer)

Statistics + actual model.
I know the PDE and train the estimate the measurement operator. There might some uncertainity in the PDE.

Example: aneurism. Inlet and outlet blood flow is not known.

Three samples:
- regular training sample for supervised learning (input/output);
- collocation points: discretization of domain;
- condition points: discretization on the boundary.

Emmpirical loss function:
- data fidelity: classical loss due to non respect of measurement
- monitor the PDE pointwise on the collocation points to assess physics satisfaction;
- monitor the PDE pointwise on the condition points to assess bcs;

Minimizing sequence is the mathematical formalization of an $\argmin$.
During training the backpropagation will follow the minimizing sequence.

If the number of collocation and condition points tend to infinity then the minimizing sequence should converge to the infimimum of the theoretical risk. This does not happen. There are pathological minimizing sequence. The empirical risk function is exactly zero but the theoretical risk is infinite. Prevent the explosition of derivative.

The Holder norm of the Neural Network is controlled by the $L^2$ norm of the parameters: Ridge regression. Incorporating the risk regression guarantees that the emprical risk will tend to the theoretical risk when the number of parameters tend to infinity.

Is this sufficient? NO, in case of imperfect modelling still some pathological cases.

Introducing Physics Inconsistency: when I regularize with Ridge and Sobolev, then outside of measurement training the physical inconsistency of the NN is bounded by the physical inconsistency of the true model.

## Physics informed machine learning and Kernel methods

Embedding of $H^s(\Omega)$ into ajn extended periodic version: norm equivalence but easier to work in periodic thanks to Fourier.

Nonlinear classicafication can be made into linear regression by lifting into a higher dimensional space. For instance redescribe data set with polynomial feature. This can be cumbersome and kernel methods help with that.

Integral/covariance operator incorporate RKHS and probability.

The Kernel can be characterized via a weak formulation (can be obtained with FEM for instance).

Takeawy:
Putting physics speeds up the learning process and increase the rate of convergence.


To avoid using weak formulation, one can use periodization and use Fourier series. 

The kernel norm becomes a bilinear form.
Complexity $(2m+1)^d \times (2m+1)^d$ where $m$ is the number of Fourier modes.

The PIKL estimator uses the physics when few data are available. Then when more data flow in it starts using them.

# From Graphs to dynamic graph NN

What is the structure of a network? What are the parts? Do they look the same?

GraphCast (2022): multi-mesh icosahedral representation of atmoshpere.

Given the graph laplacian the multiplicity of $\lambda = 0$ gives the number of connected components.

Relabelling should not change anything in the graph.


# Neural data assimilation and ocean dynamics

Chain:
Observation --> data assimilation --> physics based forecast --> post processing.

Techonology: Fortran + HPC on CPU.

Bridging different environments: is a challenge.

Resolution from 1 deg resolution to 1/60 deg resolution.

Ocean more complex than atmosphere to resolve key features.

Polar orbiting satellites measures the altitude. Very irregular sampling. This is even worse for the interior (deep sea) resolution. Only 1% of the ocean is observed (resolution 10km per 10km).

Ocean is very sparsely sampled. We want to know the state everywhere.

UQ: different trajectories are run but even one is very costly. So the distribution of the state and the variability are poorly known. AI can be a game changer.

At first AI can be used in a post-processing step to improve the resolution. AI to better parametrized unknown terms, like the closure problem in turbulence. The approximation lead to unknown term that AI can help predict.

Online learing: interaction between unknown term and the solver (guarantees that the simulation won't blow up). Requires differentiability of the solver. In practise Fortran solver is not differentiable. Then a ML emulator of the physics is used instead instead of the model to get differentiability.

Offline: dataset for unknown term and calibration.

Modules can be learned from simulation data and observations.

## Data assimilation and deep learning
Two points:
- Space-time interpolation: 
- forecasting from observations.

Observation opearator is the identity plus some mask. Optimization via variational cost:
- measurement error
- prior knowledge (based on time stepping of the model)

Many challenges:
- large state dimension
- high computational complexity
- gradients
- non convexity (the state transition is non linear)
- the dynamics is chaotic: we lose information on the initial condition.

Solutions:
- variational sequential optimization (requires gradient)
- gradient free: ensemble kalman filter, particle swarm;
- reduced rank.

## Deep learning scheme
CNN for reconstructing of state from measurements.
UNets are the state of the art. They incorporate multiscale approaches. 

Objective: exploit physical models and variational approach.

The gradient step $\alpha$ cab be replaced by a neural network $\mathcal{R}$. Then other parameters are $\Phi, \mathbf{H}$. This can be done in a supervised settings.

The paradoxical thing is that the model free problem performs better. The guess is that the prior needs to be adapted to resolution (sparse sampling).

# Probabilistic weather forecast through ML (Nvidia)

Neural operator are unstable because the spherical geometry is not respected.

Symmetry matters. Shallow water is symmetric with respect to rotation on the sphere. Bottom topologies break symmetries in a weak sense.

Cheap spherical fourier convolution.

Models drived from ML are cheap to inference: large ensemble are now possible.

Perturbed initial conditions: gaussian noise, do few steps, take the difference the pertubed and unbertubed trajectories, take the perturbation that create maximal deviation and sample from that. 

Images are strongly correlated but physical values are not.

The memory is dominated by the activation functions. Makani (check github) handles the distributed architecture.

## Learned lessons
The best way for a better model is through scaling (dataset and architecture).
Iterating on a supercomputing is very difficult.

Aim for insight first: add visualization in the loop (it pay off especilly given the iteration on a supercomputer)

Hidden failure modes (adam does not track momenta in complex domains). Do no trust the framework.

Different FFT in the backgroup (for instance imaginary part of 0 zero mode is not set to zero for a real signal).

Not deterministic in a distributed framewrok for the gradient.

Iteration time should be kept down but don't neglect scaling.


# Online training of deep surrogate models

Foundamental problem : I/O big data.

Solution don't write everything at once but iteratively update statistical quantities of interest (Melissa code). Iterative statistics. Quantize on the fly, sobol indexes (sensitivity measures) on the fly etc...

How to train substitution model at scale?

Deep Surrogates: NN to approximate solver.

Two ways:
- Autoregressive $u_{t} = f_\theta(u_{t-1})$
- Direct $u_t = f_\theta(u_0, t)$

NN can compress massively: visualization becomes possible.

Exploration of parametric space.

- no data required: PINNs.
- training data required: Neural operator, GNN, Vision Tranformer

Training with no enough data. Then do more epochs.

Split batch on multiple GPUs. Gradient descent has a sum: sum is commutative. Then one can achieve sublinear complexity via all-reduce ($\log p$ with $p$ number of GPUs).

# SCIML persepectives on solving control problems

Optimal control and RL:
In discrete dynamical systems may be represeted by a Markov Decision Process

In policy iterative we converge to the optimal value in finite steps. Many policy are explored.

Hybrid way: value V and policy function are discovered at the same time.

- HJB can have an infinity of generalized solution in continuous spaces of the state.
- The value function is in general non-smooth.

Viscosity solution is the tool to distinguish the value function from other solution of the HJB equation.

We take smooth test functions that approximate from below and from above the (non-smooth) solution. The test functin touches the non-smooth solution at kinks.

When HJB is discretized via finite difference we obtain a markov decision process that can be solved via dynamic programming. Same story for FEM.
But the problem is the curse of dimensionality.

Learning dynamical systems by using neural ODE with a bayesan touch to account for stocasticity.

# Industrialized ML

Use GP with kernel taking graphs (that described FEM meshes).

Continuous Node attribute: nodes position.

New graph: find an empirical law distribution from the Graph

# Foundation model for PDEs

Catch of classical numerical methods:
- High cost;
- many query problems: UQ, Design, Inverse problems.
- Chaotic systems: different initial conditions produce widely different trajectories.

Monte carlo can be used to generate useful statistics.

Entire physics in any case is not known. But with still have data.

Operator learning task: given a dsitribution of the data find the push-forward of the distribution through the solution operator.

DNN: input and output are vectors:
NO (neural operator): composition of hidden layer obtained via kernel integral operators.

By itself NO cannot be implemented in a computer. Scalability is a problem. If for an axis one has $N$ points then an itegral operator is  of size $(N^d)^2$ where $\Omega \subset \mathbb{R}^d$.

Translation invariance makes the integral operator into a convolution. For cartesian grids using FFT one scale with $N\log N$.

New paradigm: local convolution but nonlocal activation. Local nonlinearities induce high frequency and ugly artifact. To avois this, the nonlinearity is made nnolocal.

CNO (convolution neural operator) are structure preserving operators. Error remains low for different resolutions.

Nonlinear Kernel: use attention to track iteraction between different points. Scales badly as for every point one check $N^2$ for every axis.

One can use patching to reduce the complexity.

Problem: collapse of the mean.

What is meaninful in physics is statistical quantity of interest.

Stocastic differential equation: allow to add noise in diffusion models. Reverse SDE denoises: the denoiser minimizes the level of noise.