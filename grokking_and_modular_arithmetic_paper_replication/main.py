# %% Setup
import torch as t
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import os
import sys

import plotly.express as px
import plotly.graph_objects as go

from functools import *
import gdown
from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int
from tqdm import tqdm

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_grokking_and_modular_arithmetic"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

root = (section_dir / 'Grokking' / 'saved_runs').resolve()
large_root = (section_dir / 'Grokking' / 'large_files').resolve()

from part5_grokking_and_modular_arithmetic.my_utils import *
import part5_grokking_and_modular_arithmetic.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %% Part 1

# %% cfg
p = 113

cfg = HookedTransformerConfig(
    n_layers = 1,
    d_vocab = p+1,
    d_model = 128,
    d_mlp = 4 * 128,
    n_heads = 4,
    d_head = 128 // 4,
    n_ctx = 3,
    act_fn = "relu",
    normalization_type = None,
    device = device
)

model = HookedTransformer(cfg)
# %% download data
os.chdir(section_dir)
if not large_root.exists(): 
    !git clone https://github.com/neelnanda-io/Grokking.git
    os.mkdir(large_root)

full_run_data_path = (large_root / "full_run_data.pth").resolve()
if not full_run_data_path.exists():
    url = "https://drive.google.com/uc?id=12pmgxpTHLDzSNMbMCuAMXP1lE_XiCQRy"
    output = str(full_run_data_path)
    gdown.download(url, output)

# %%
full_run_data = t.load(full_run_data_path, map_location=t.device('cpu'))
state_dict = full_run_data["state_dicts"][400]

model = load_in_state_dict(model, state_dict)

# %% plot loss lines
lines(
    lines_list=[
        full_run_data['train_losses'][::10], 
        full_run_data['test_losses']
    ], 
    labels=['train loss', 'test loss'], 
    title='Grokking Training Curve', 
    x=np.arange(5000)*10,
    xaxis='Epoch',
    yaxis='Loss',
    log_y=True
)

# %% Helper variables
# Helper variables
W_O = model.W_O[0]
W_K = model.W_K[0]
W_Q = model.W_Q[0]
W_V = model.W_V[0]
W_in = model.W_in[0]
W_out = model.W_out[0]
W_pos = model.W_pos
W_E = model.W_E[:-1]
final_pos_resid_initial = model.W_E[-1] + W_pos[2]
W_U = model.W_U[:, :-1]

print('W_O  ', tuple(W_O.shape))
print('W_K  ', tuple(W_K.shape))
print('W_Q  ', tuple(W_Q.shape))
print('W_V  ', tuple(W_V.shape))
print('W_in ', tuple(W_in.shape))
print('W_out', tuple(W_out.shape))
print('W_pos', tuple(W_pos.shape))
print('W_E  ', tuple(W_E.shape))
print('W_U  ', tuple(W_U.shape))

# %% Run on all data
all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([fn(i, j) for i, j, _ in all_data]).to(device)
original_logits, cache = model.run_with_cache(all_data)
# Final position only, also remove the logits for `=`
original_logits = original_logits[:, -1, :-1]
original_loss = cross_entropy_high_precision(original_logits, labels)
print(f"Original loss: {original_loss.item()}")

# %% extract key activations

attn_mat = cache[utils.get_act_name("pattern",0)][...,-1,:]
neuron_acts_post = cache[utils.get_act_name("post",0)][...,-1,:]
neuron_acts_pre = cache[utils.get_act_name("pre",0)][...,-1,:]

assert attn_mat.shape == (p*p, cfg.n_heads, 3)
assert neuron_acts_post.shape == (p*p, cfg.d_mlp)
assert neuron_acts_pre.shape == (p*p, cfg.d_mlp)

# %% answering initial questions
# Get the first three positional embedding vectors
W_pos_x, W_pos_y, W_pos_equals = W_pos

# Look at the difference between positional embeddings; show they are symmetric
def compare_tensors(v, w):
    return ((v-w).pow(2).sum()/v.pow(2).sum().sqrt()/w.pow(2).sum().sqrt()).item()
print('Difference in position embeddings', compare_tensors(W_pos_x, W_pos_y))
print('Cosine similarity of position embeddings', t.cosine_similarity(W_pos_x, W_pos_y, dim=0).item())

# Compare N(x, y) and N(y, x)
neuron_acts_square = neuron_acts.reshape(p, p, d_mlp)
print('Difference in neuron activations for (x,y) and (y,x): {.2f}'.format(
    compare_tensors(
        neuron_acts_square, 
        einops.rearrange(neuron_acts_square, "x y d_mlp -> y x d_mlp")
    )
))
# %%
imshow(attn_mat.mean(0), xaxis='Position', yaxis='Head', title='Average Attention by source position and head', text_auto=".3f")

# %% define the effective weight matrices

W_logit = W_out @ W_U
W_neur = W_E @ W_V @ W_O @ W_in

W_QK = W_Q @ W_K.transpose(-1, -2)
W_attn = final_pos_resid_initial @ W_QK @ W_E.T / (cfg.d_head ** 0.5)

assert W_logit.shape == (cfg.d_mlp, cfg.d_vocab - 1)
assert W_neur.shape == (cfg.n_heads, cfg.d_vocab - 1, cfg.d_mlp)
assert W_attn.shape == (cfg.n_heads, cfg.d_vocab - 1)

# %% attention heatmap
attn_mat = attn_mat[:, :, :2]
# Note, we ignore attn from 2 -> 2

attn_mat_sq = einops.rearrange(attn_mat, "(x y) head seq -> x y head seq", x=p)
# We rearranged attn_mat, so the first two dims represent (x, y) in modular arithmetic equation

inputs_heatmap(
    attn_mat_sq[..., 0], 
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)
# %% neurons heatmap
neuron_acts_post_sq = einops.rearrange(neuron_acts_post, "(x y) d_mlp -> x y d_mlp", x=p)
neuron_acts_pre_sq = einops.rearrange(neuron_acts_pre, "(x y) d_mlp -> x y d_mlp", x=p)
# We rearranged activations, so the first two dims represent (x, y) in modular arithmetic equation

top_k = 3
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)

# %% W_neur
top_k = 5
animate_multi_lines(
    W_neur[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)'
)
# %% W_attn
lines(
    W_attn,
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis='Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head'
)

# %% 1D Fourier basis
def make_fourier_basis(p: int) -> Tuple[Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g. 
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    # SOLUTION
    # Define a grid for the Fourier basis vecs (we'll normalize them all at the end)
    # Note, the first vector is just the constant wave
    fourier_basis = t.ones(p, p)
    fourier_basis_names = ['Const']
    for i in range(1, p // 2 + 1):
        # Define each of the cos and sin terms
        fourier_basis[2*i-1] = t.cos(2*t.pi*t.arange(p)*i/p)
        fourier_basis[2*i] = t.sin(2*t.pi*t.arange(p)*i/p)
        fourier_basis_names.extend([f'cos {i}', f'sin {i}'])
    # Normalize vectors, and return them
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis.to(device), fourier_basis_names

tests.test_make_fourier_basis(make_fourier_basis)

# %%
fourier_basis, fourier_basis_names = make_fourier_basis(p)

animate_lines(
    fourier_basis, 
    snapshot_index=fourier_basis_names, 
    snapshot='Fourier Component', 
    title='Graphs of Fourier Components (Use Slider)'
)

imshow(fourier_basis @ fourier_basis.T)

# %% fft1d
def fft1d(x: t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.

    x.shape = (..., p)
    '''
    return x @ fourier_basis.T

tests.test_fft1d(fft1d)

# %%
v = sum([
    fourier_basis[4],
    fourier_basis[15]/5,
    fourier_basis[67]/10
])

line(v, xaxis='Vocab basis', title='Example periodic function')
line(fft1d(v), xaxis='Fourier Basis', title='Fourier Transform of example function', hover=fourier_basis_names)

# %% fourier_2d_basis_term
def fourier_2d_basis_term(i: int, j: int) -> Float[Tensor, "p p"]:
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    fb = make_fourier_basis(p)[0]
    #return fb[i] @ fb[j].T
    return einops.einsum(fb[i], fb[j], "i, j -> i j")

tests.test_fourier_2d_basis_term(fourier_2d_basis_term)

x_term = 4
y_term = 6

inputs_heatmap(
    fourier_2d_basis_term(x_term, y_term).T,
    title=f"2D Fourier Basis term {fourier_basis_names[x_term]}x {fourier_basis_names[y_term]}y"
)

# %% fft2d
def fft2d(tensor: t.Tensor) -> t.Tensor:
    '''
    Retuns the components of `tensor` in the 2D Fourier basis.

    Asumes that the input has shape `(p, p, ...)`, where the
    last dimensions (if present) are the batch dims.
    Output has the same shape as the input.
    '''
    return einops.einsum(
        tensor, fourier_basis, fourier_basis,
        "px py ..., i px, j py -> i j ..."
    )    

tests.test_fft2d(fft2d)

# %%
example_fn = sum([
    fourier_2d_basis_term(4, 6), 
    fourier_2d_basis_term(14, 46) / 3,
    fourier_2d_basis_term(97, 100) / 6
])

inputs_heatmap(example_fn.T, title=f"Example periodic function")

imshow_fourier(
    fft2d(example_fn),
    title='Example periodic function in 2D Fourier basis'
)

# %% heatmap Fourier attention
inputs_heatmap(
    attn_mat[..., 0], 
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)

# Apply Fourier transformation
attn_mat_fourier_basis = fft2d(attn_mat_sq)

# Plot results
imshow_fourier(
    attn_mat_fourier_basis[..., 0], 
    title=f'Attention score for heads at position 0, in Fourier basis',
    animation_frame=2,
    animation_name='head'
)

# %%
top_k = 40
inputs_heatmap(
    neuron_acts_post[:, :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)

neuron_acts_post_fourier_basis = fft2d(neuron_acts_post_sq)

imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)

# %%
top_k = 5
animate_multi_lines(
    W_neur[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn)'
)

# %%
def fft1d_given_dim(tensor: t.Tensor, dim: int) -> t.Tensor:
    '''
    Performs 1D FFT along the given dimension (not necessarily the last one).
    '''
    return fft1d(tensor.transpose(dim, -1)).transpose(dim, -1)


W_neur_fourier = fft1d_given_dim(W_neur, dim=1)

top_k = 5
animate_multi_lines(
    W_neur_fourier[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Fourier component', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    hover=fourier_basis_names,
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn), in Fourier basis'
)

# %%

lines(
    fft1d(W_attn), 
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token', 
    yaxis = 'Contribution to attn score',
    title=f'Contribution to attn score (pre-softmax) for each head, in Fourier Basis', 
    hover=fourier_basis_names
)

# %% Part 2

# %%
line(
    (fourier_basis @ W_E).pow(2).sum(1), 
    hover=fourier_basis_names,
    title='Norm of embedding of each Fourier Component',
    xaxis='Fourier Component',
    yaxis='Norm'
)
imshow_div(fourier_basis @ W_E)

# %%
top_k = 5
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
# %% mean squared coefficient

neuron_acts_centered = neuron_acts_post_sq - neuron_acts_post_sq.mean((0, 1), keepdim=True)
# Take 2D Fourier transform
neuron_acts_centered_fourier = fft2d(neuron_acts_centered)

imshow_fourier(
    neuron_acts_centered_fourier.pow(2).mean(-1),
    title=f"Norms of 2D Fourier components of centered neuron activations",
)

# %% fit lin reg
from sklearn.linear_model import LinearRegression

# Choose a particular frequency, and get the corresponding cosine basis vector
k = 42
idx = 2 * k - 1
vec = fourier_basis[idx]

# Get ReLU function values
relu_func_values = F.relu(0.5 * (p ** -0.5) + vec[None, :] + vec[:, None])

# Get terms we'll be using to approximate it
# Note we're including the constant term here
data = t.stack([
    fourier_2d_basis_term(i, j)
    for (i, j) in [(0, 0), (idx, 0), (0, idx), (idx, idx)]
], dim=-1)

# Reshape, and convert to numpy
data = utils.to_numpy(data.reshape(p*p, 4))
relu_func_values = utils.to_numpy(relu_func_values.flatten())

# Fit a linear model (we don't need intercept because we have const Fourier basis term)
reg = LinearRegression(fit_intercept=False).fit(data, relu_func_values)
coefs = reg.coef_
eqn = "ReLU(0.5 + cos(wx) + cos(wy) ≈ {:.3f}*const + {:.3f}*cos(wx) + {:.3f}*cos(wy) + {:.3f}*cos(wx)cos(wy)".format(*coefs)
r2 = reg.score(data, relu_func_values)
print(eqn)
print("")
print(f"r2: {r2:.3f}")

# Run the regression again, but without the quadratic term
data = data[:, :3]
reg = LinearRegression().fit(data, relu_func_values)
coefs = reg.coef_
bias = reg.intercept_
r2 = reg.score(data, relu_func_values)
print(f"r2 (no quadratic term): {r2:.3f}")

# %%
def arrange_by_2d_freqs(tensor):
    '''
    Takes a tensor of shape (p, p, ...) and returns a tensor of shape
    (p//2 - 1, 3, 3, ...) representing the Fourier coefficients sorted by
    frequency (each slice contains const, linear and quadratic terms).

    In other words, if the first two dimensions of the original tensor
    correspond to indexing by 2D Fourier frequencies as follows:

        1           cos(w_1*x)            sin(w_1*x)           ...
        cos(w_1*y)  cos(w_1*x)cos(w_1*y)  sin(w_1*x)cos(w_1*y) ...
        sin(w_1*y)  cos(w_1*x)sin(w_1*y)  sin(w_1*x)sin(w_1*y) ...
        cos(w_2*y)  cos(w_1*x)cos(w_2*y)  sin(w_1*x)cos(w_2*y) ...
        ...

    Then the (k-1)-th slice of the new tensor are the terms corresponding to 
    the following 2D Fourier frequencies:

        1           cos(w_k*x)            sin(w_k*x)           ...
        cos(w_k*y)  cos(w_k*x)cos(w_k*y)  sin(w_k*x)cos(w_k*y) ...
        sin(w_k*y)  cos(w_k*x)sin(w_k*y)  sin(w_k*x)sin(w_k*y) ...

    for k = 1, 2, ..., p//2.

    Note we omit the constant term, i.e. the 0th slice has frequency k=1.
    '''
    idx_2d_y_all = []
    idx_2d_x_all = []
    for freq in range(1, p//2):
        idx_1d = [0, 2*freq-1, 2*freq]
        idx_2d_x_all.append([idx_1d for _ in range(3)])
        idx_2d_y_all.append([[i]*3 for i in idx_1d])
    return tensor[idx_2d_y_all, idx_2d_x_all]


def find_neuron_freqs(
    fourier_neuron_acts: Float[Tensor, "p p d_mlp"]
) -> Tuple[Float[Tensor, "d_mlp"], Float[Tensor, "d_mlp"]]:
    '''
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`, 
    containing the frequencies that explain the most variance of each 
    neuron and the fraction of variance explained, respectively.
    '''
    fourier_neuron_acts_by_freq = arrange_by_2d_freqs(fourier_neuron_acts)
    assert fourier_neuron_acts_by_freq.shape == (p//2-1, 3, 3, d_mlp)

    sum_squares = fourier_neuron_acts_by_freq.pow(2).sum(dim=(1,2))
    total_var = sum_squares.sum(dim = 0, keepdim=True)
    proportion = sum_squares/total_var
    neuron_frac_explained, neuron_freqs = proportion.max(dim = 0)
    neuron_freqs += 1
    return (neuron_freqs, neuron_frac_explained)


neuron_freqs, neuron_frac_explained = find_neuron_freqs(neuron_acts_centered_fourier)
key_freqs, neuron_freq_counts = t.unique(neuron_freqs, return_counts=True)


assert key_freqs.tolist() == [14, 35, 41, 42, 52]

fraction_of_activations_positive_at_posn2 = (cache['pre', 0][:, -1] > 0).float().mean(0)

scatter(
    x=neuron_freqs, 
    y=neuron_frac_explained,
    xaxis="Neuron frequency", 
    yaxis="Frac explained", 
    colorbar_title="Frac positive",
    title="Fraction of neuron activations explained by key freq",
    color=utils.to_numpy(fraction_of_activations_positive_at_posn2)
)

# %%
# To represent that they are in a special sixth cluster, we set the frequency of these neurons to -1
neuron_freqs[neuron_frac_explained < 0.85] = -1.
key_freqs_plus = t.concatenate([key_freqs, -key_freqs.new_ones((1,))])

for i, k in enumerate(key_freqs_plus):
    print(f'Cluster {i}: freq k={k}, {(neuron_freqs==k).sum()} neurons')

# %%
fourier_norms_in_each_cluster = []
for freq in key_freqs:
    fourier_norms_in_each_cluster.append(
        einops.reduce(
            neuron_acts_centered_fourier.pow(2)[..., neuron_freqs==freq], 
            'batch_y batch_x neuron -> batch_y batch_x', 
            'mean'
        )
    )

imshow_fourier(
    t.stack(fourier_norms_in_each_cluster), 
    title=f'Norm of 2D Fourier components of neuron activations in each cluster',
    facet_col=0,
    facet_labels=[f"Freq={freq}" for freq in key_freqs]
)
# %%
def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''
    # Get tensor of components of each vector in v-direction
    components_in_v_dir = einops.einsum(
        batch_vecs, v,
        "n ..., n -> ..."
    )

    # Use these components as coefficients of v in our projections
    return einops.einsum(
        components_in_v_dir, v,
        "..., n -> n ..."
    )

tests.test_project_onto_direction(project_onto_direction)

# %%
def project_onto_frequency(batch_vecs: t.Tensor, freq: int) -> t.Tensor:
    '''
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.

    batch_vecs.shape = (p**2, ...)
    '''
    assert batch_vecs.shape[0] == p**2
    result = t.zeros_like(batch_vecs)
    indices = [0, 2*freq-1, 2*freq]
    bases = [(i,j) for i in indices for j in indices]
    for (i,j) in bases:
        fourier_2d = fourier_2d_basis_term(i,j).reshape(p**2)
        result += project_onto_direction(batch_vecs, fourier_2d)
    return result

tests.test_project_onto_frequency(project_onto_frequency)

# %%
logits_in_freqs = []

for freq in key_freqs:

    # Get all neuron activations corresponding to this frequency
    filtered_neuron_acts = neuron_acts_post[:, neuron_freqs==freq]

    # Project onto const/linear/quadratic terms in 2D Fourier basis
    filtered_neuron_acts_in_freq = project_onto_frequency(filtered_neuron_acts, freq)

    # Calcluate new logits, from these filtered neuron activations
    logits_in_freq = filtered_neuron_acts_in_freq @ W_logit[neuron_freqs==freq]

    logits_in_freqs.append(logits_in_freq)

# We add on neurons in the always firing cluster, unfiltered
logits_always_firing = neuron_acts_post[:, neuron_freqs==-1] @ W_logit[neuron_freqs==-1]
logits_in_freqs.append(logits_always_firing)

# Print new losses
print('Loss with neuron activations ONLY in key freq (inclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Loss with neuron activations ONLY in key freq (exclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs[:-1]), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Original loss\n{:.6e}'.format(original_loss))


# %%
print('Loss with neuron activations excluding none:     {:.9f}'.format(original_loss.item()))
for c, freq in enumerate(key_freqs_plus):
    print('Loss with neuron activations excluding freq={}:  {:.9f}'.format(
        freq, 
        test_logits(
            sum(logits_in_freqs) - logits_in_freqs[c], 
            bias_correction=True, 
            original_logits=original_logits
        )
    ))

# %%
imshow_fourier(
    einops.reduce(neuron_acts_centered_fourier.pow(2), 'y x neuron -> y x', 'mean'), 
    title='Norm of Fourier Components of Neuron Acts'
)

# Rearrange logits, so the first two dims represent (x, y) in modular arithmetic equation
original_logits_sq = einops.rearrange(original_logits, "(x y) z -> x y z", x=p)
original_logits_fourier = fft2d(original_logits_sq)

imshow_fourier(
    einops.reduce(original_logits_fourier.pow(2), 'y x z -> y x', 'mean'), 
    title='Norm of Fourier Components of Logits'
)
# %%
def get_trig_sum_directions(k: int) -> Tuple[Float[Tensor, "p p"], Float[Tensor, "p p"]]:
    '''
    Given frequency k, returns the normalized vectors in the 2D Fourier basis 
    representing the directions:

        cos(ω_k * (x + y))
        sin(ω_k * (x + y))

    respectively.
    '''
    cosx_cosy_direction = fourier_2d_basis_term(2*k-1, 2*k-1)
    sinx_siny_direction = fourier_2d_basis_term(2*k, 2*k)
    sinx_cosy_direction = fourier_2d_basis_term(2*k, 2*k-1)
    cosx_siny_direction = fourier_2d_basis_term(2*k-1, 2*k)
    cosxy = cosx_cosy_direction - sinx_siny_direction
    sinxy = sinx_cosy_direction + cosx_siny_direction
    cosxy = cosxy / np.sqrt(2)
    sinxy = sinxy / np.sqrt(2)
    return (cosxy, sinxy)

tests.test_get_trig_sum_directions(get_trig_sum_directions)

# %%
trig_logits = []

for k in key_freqs:

    cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(k)

    cos_xplusy_projection = project_onto_direction(
        original_logits,
        cos_xplusy_direction.flatten()
    )

    sin_xplusy_projection = project_onto_direction(
        original_logits,
        sin_xplusy_direction.flatten()
    )

    trig_logits.extend([cos_xplusy_projection, sin_xplusy_projection])

trig_logits = sum(trig_logits)

print(f'Loss with just x+y components: {test_logits(trig_logits, True, original_logits):.4e}')
print(f"Original Loss: {original_loss:.4e}")

# %%
US = W_logit @ fourier_basis.T

imshow_div(
    US,
    x=fourier_basis_names,
    yaxis='Neuron index',
    title='W_logit in the Fourier Basis',
    height=800,
    width=600
)
# %%
US_sorted = t.concatenate([
    US[neuron_freqs==freq] for freq in key_freqs_plus
])
hline_positions = np.cumsum([(neuron_freqs == freq).sum().item() for freq in key_freqs]).tolist() + [cfg.d_mlp]

imshow_div(
    US_sorted,
    x=fourier_basis_names, 
    yaxis='Neuron',
    title='W_logit in the Fourier Basis (rearranged by neuron cluster)',
    hline_positions = hline_positions,
    hline_labels = [f"Cluster: {freq=}" for freq in key_freqs.tolist()] + ["No freq"],
    height=800,
    width=600
)
# %%
cos_components = []
sin_components = []

for k in key_freqs:
    σu_sin = US[:, 2*k]
    σu_cos = US[:, 2*k-1]

    logits_in_cos_dir = neuron_acts_post_sq @ σu_cos
    logits_in_sin_dir = neuron_acts_post_sq @ σu_sin

    cos_components.append(fft2d(logits_in_cos_dir))
    sin_components.append(fft2d(logits_in_sin_dir))

for title, components in zip(['Cosine', 'Sine'], [cos_components, sin_components]):
    imshow_fourier(
        t.stack(components),
        title=f'{title} components of neuron activations in Fourier basis',
        animation_frame=0,
        animation_name="Frequency",
        animation_labels=key_freqs.tolist()
    )

# %%
