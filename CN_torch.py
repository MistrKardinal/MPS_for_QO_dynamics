"""Self-contained two-site Crank-Nicolson MPS solver (PyTorch).

The contractions, reduced linear-system builder, sweeps, SVD compression,
and evolution diagnostics are implemented within this module.
"""

import torch
import numpy as np
import time
import pickle
import warnings
from tqdm import tqdm


def mps_scalar_multiplication(mps_1, mps_2):
    """Contract two MPS networks to obtain the complex inner product <mps_1|mps_2>."""
    tensor = torch.ones((1, 1), dtype=torch.promote_types(mps_1[0].dtype, mps_2[0].dtype), device=mps_1[0].device)
    for i in range(len(mps_1)):
        tensor = torch.einsum('ij, ikl, jkg -> lg', tensor, torch.conj(mps_1[i]), mps_2[i])
    return tensor.reshape(-1)[0]


def compute_norm(mps):
    """Calculate the Euclidean norm of an MPS without reconstructing its full vector."""
    return torch.sqrt(torch.clamp(mps_scalar_multiplication(mps, mps).real, min=0))


def right_canonicalize(mps):
    """Move the MPS orthogonality centre to the left via QR decompositions from right to left."""
    rc_mps = mps.copy()
    for i in range(1, len(mps)):
        shape = rc_mps[-i].shape
        matrix = torch.conj(rc_mps[-i].reshape(shape[0], shape[1]*shape[2])).T
        q, r = torch.linalg.qr(matrix, mode='reduced')
        rc_mps[-i] = torch.conj(q).T.reshape(q.shape[1], shape[1], shape[2])
        r_conj = torch.conj(r).T
        rc_mps[-i-1] = torch.einsum('ijk, kl -> ijl', rc_mps[-i-1], r_conj)
    return rc_mps


def left_canonical_MPS(mps):
    """Transform the tensor train to a left-canonical MPS by sequential QR factorizations."""
    mps_copy = mps.copy()
    left_canonical_mps = []
    tensor = mps_copy[0]
    for i in range(len(mps_copy)-1):
        sh = tensor.shape
        q, r = torch.linalg.qr(tensor.reshape(sh[0]*sh[1], sh[2]), mode='reduced')
        left_canonical_mps.append(q.reshape(sh[0], sh[1], -1))
        tensor = torch.tensordot(r, mps_copy[i+1], dims=1)
    left_canonical_mps.append(tensor)
    return left_canonical_mps


def mps_compression(mps, eps=1e-6, bond_limit=500):
    """Reduce the virtual-bond dimensions using truncated singular-value decompositions.
    
    Parameters:
        mps: MPS tensor list with shape (left_bond, physical, right_bond).
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
    
    Returns:
        New list of compressed MPS cores."""
    if not isinstance(mps, list):
        raise TypeError('Аргумент должен быть списком')
    mps = left_canonical_MPS(mps)
    compressed_mps = []
    tensor = mps[-1]
    for i in range(len(mps)-1):
        sh = tensor.shape
        u, s, vh = torch.linalg.svd(tensor.reshape(sh[0], sh[1]*sh[2]), full_matrices=False)
        s2 = s**2
        relative_tail2 = torch.cumsum(s2.flip(0), 0).flip(0) / s2.sum().clamp_min(torch.finfo(s.dtype).tiny)
        r = int((relative_tail2 > eps**2).sum().item())
        bond = min(max(r, 1), bond_limit)
        compressed_mps.append(vh[:bond].reshape(bond, sh[1], sh[2]))
        tensor = torch.tensordot(mps[-i-2], u[:, :bond]*s[:bond], dims=1)
    compressed_mps.append(tensor)
    return compressed_mps[::-1]


def mpo_sum(mpo_one, mpo_two):
    """Construct the sum of two MPOs by combining their virtual-bond spaces."""
    common_tensor = [torch.cat([mpo_one[0], mpo_two[0]], dim=3)]
    for i in range(1, len(mpo_one)-1):
        a, b = mpo_one[i], mpo_two[i]
        dtype = torch.promote_types(a.dtype, b.dtype)
        a, b = a.to(dtype), b.to(dtype)
        zero_arr1 = torch.zeros((b.shape[0], 2, 2, a.shape[3]), dtype=dtype, device=a.device)
        zero_arr2 = torch.zeros((a.shape[0], 2, 2, b.shape[3]), dtype=dtype, device=a.device)
        ff1 = torch.cat([a, zero_arr1], dim=0)
        ff2 = torch.cat([zero_arr2, b], dim=0)
        common_tensor.append(torch.cat([ff1, ff2], dim=3))
    common_tensor.append(torch.cat([mpo_one[-1], mpo_two[-1]], dim=0))
    return common_tensor


def one_mpo(n, dtype=torch.complex128, device='cpu'):
    """Construct an identity MPO for n binary physical sites."""
    return [torch.eye(2, dtype=dtype, device=device).reshape(1, 2, 2, 1) for _ in range(n)]


def mpo_mul_num(mpo, num):
    """Multiply an MPO by a scalar by rescaling its first tensor only."""
    mpo_ch = mpo.copy()
    mpo_ch[0] = mpo_ch[0] * num
    return mpo_ch


def mps_mul_num(mps, num):
    """Multiply an MPS by a scalar by rescaling its first tensor only."""
    mps_ch = mps.copy()
    mps_ch[0] = mps_ch[0] * num
    return mps_ch


def mps_sum(mps_one, mps_two):
    """Represent the sum of two MPS states through a direct sum of virtual bonds."""
    result = [torch.cat([mps_one[0], mps_two[0]], dim=2)]
    for i in range(1, len(mps_one)-1):
        a, b = mps_one[i], mps_two[i]
        dtype = torch.promote_types(a.dtype, b.dtype)
        a, b = a.to(dtype), b.to(dtype)
        upper = torch.cat([a, torch.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=dtype, device=a.device)], dim=2)
        lower = torch.cat([torch.zeros((b.shape[0], b.shape[1], a.shape[2]), dtype=dtype, device=a.device), b], dim=2)
        result.append(torch.cat([upper, lower], dim=0))
    result.append(torch.cat([mps_one[-1], mps_two[-1]], dim=0))
    return result


def mpo_x_mps(mpo, mps):
    # Умножение по физическому индексу через batched matmul (вместо einsum).
    """Apply an MPO to an MPS, without automatically compressing the resulting bonds."""
    new_mps = []
    for i in range(len(mps)):
        w, v = mpo[i], mps[i]
        wl, d_out, d_in, wr = w.shape
        vl, _, vr = v.shape
        matrix_w = w.permute(1, 0, 3, 2).reshape(d_out, wl*wr, d_in)
        matrix_v = v.permute(1, 0, 2).reshape(d_in, vl*vr)
        tensor = torch.matmul(matrix_w, matrix_v)
        tensor = tensor.reshape(d_out, wl, wr, vl, vr).permute(1, 3, 0, 2, 4)
        new_mps.append(tensor.reshape(wl*vl, d_out, wr*vr))
    return new_mps


def make_cn_operators(mpo, delta_t):
    # Строится единожды на весь интервал эволюции.
    # Сохраняем точность входного MPO: complex64 не переводим принудительно в complex128.
    """Build the Crank–Nicolson left operator A = I + (i dt/2) H as an MPO."""
    dtype = torch.promote_types(mpo[0].dtype, torch.complex64)
    h = [tensor.to(dtype) for tensor in mpo]
    identity_mpo = one_mpo(len(mpo), dtype=dtype, device=mpo[0].device)
    mpo_ls = mpo_sum(identity_mpo, mpo_mul_num(h, 1j*delta_t/2))
    return mpo_ls


def compute_cn_right_part(hamiltonian_mpo, state, delta_t, eps=1e-9, bond_limit=1000, h_state=None):
    # (I - i*dt*H/2)|psi> = |psi> - i*dt/2 H|psi>.
    # Не требуется формировать дополнительный MPO для правой части.
    """Build b = (I - i dt H/2) |psi> and truncate the MPS representation."""
    if h_state is None:
        h_state = mpo_x_mps(hamiltonian_mpo, state)
    right_part = mps_sum(state, mps_mul_num(h_state, -1j*delta_t/2))
    return mps_compression(right_part, eps=eps, bond_limit=bond_limit)


def compute_mps_layer_from_left_to_right(layer, tensors, optimize=True):
    # L'_{r_b,r_x} = sum_{l_b,l_x,s} L_{l_b,l_x} b_{l_b,s,r_b} conj(x_{l_x,s,r_x})
    """Extend a left boundary environment for an MPS overlap by one physical site."""
    b, x = tensors
    tmp = torch.matmul(layer.T, b.reshape(b.shape[0], -1))
    tmp = tmp.reshape(layer.shape[1], b.shape[1], b.shape[2])
    return torch.matmul(tmp.permute(2, 0, 1).reshape(b.shape[2], -1), x.reshape(x.shape[0]*x.shape[1], x.shape[2]))


def compute_mps_layer_from_right_to_left(layer, tensors, optimize=True):
    """Extend a right boundary environment for an MPS overlap by one physical site."""
    b, x = tensors
    tmp = torch.matmul(b.reshape(-1, b.shape[2]), layer)
    tmp = tmp.reshape(b.shape[0], b.shape[1], x.shape[2])
    return torch.matmul(tmp.reshape(b.shape[0], -1), x.permute(1, 2, 0).reshape(-1, x.shape[0]))


def compute_mpo_layer_from_left_to_right(layer, tensors, optimize=True):
    """Update the left effective-operator environment by contracting one MPO/MPS site."""
    x, w, y = tensors
    # Последовательные матричные умножения по общим индексам.
    tmp = torch.matmul(layer.permute(1, 2, 0).reshape(-1, x.shape[0]), x.reshape(x.shape[0], -1))
    tmp = tmp.reshape(layer.shape[1], layer.shape[2], x.shape[1], x.shape[2]).permute(3, 2, 0, 1)
    tmp = torch.matmul(tmp.permute(0, 3, 1, 2).reshape(-1, w.shape[0]*w.shape[1]), w.permute(1, 0, 2, 3).reshape(w.shape[0]*w.shape[1], -1))
    tmp = tmp.reshape(x.shape[2], y.shape[0], w.shape[2], w.shape[3]).permute(0, 3, 2, 1)
    tmp = torch.matmul(tmp.reshape(x.shape[2]*w.shape[3], -1), y.permute(1, 0, 2).reshape(-1, y.shape[2]))
    return tmp.reshape(x.shape[2], w.shape[3], y.shape[2])


def compute_mpo_layer_from_right_to_left(layer, tensors, optimize=True):
    """Update the right effective-operator environment by contracting one MPO/MPS site."""
    x, w, y = tensors
    tmp = torch.matmul(x.reshape(-1, x.shape[2]), layer.reshape(layer.shape[0], -1))
    tmp = tmp.reshape(x.shape[0], x.shape[1], layer.shape[1], layer.shape[2])
    tmp = torch.matmul(tmp.permute(0, 3, 1, 2).reshape(-1, w.shape[1]*w.shape[3]), w.permute(1, 3, 0, 2).reshape(w.shape[1]*w.shape[3], -1))
    tmp = tmp.reshape(x.shape[0], y.shape[2], w.shape[0], w.shape[2]).permute(0, 2, 3, 1)
    tmp = torch.matmul(tmp.reshape(x.shape[0]*w.shape[0], -1), y.permute(1, 2, 0).reshape(-1, y.shape[0]))
    return tmp.reshape(x.shape[0], w.shape[0], y.shape[0])


def compute_reduced_A(left_layers, right_layers, mpo_1, mpo_2, optimize=True):
    """Build the two-site effective matrix of the linear system from environments and two MPO cores."""
    tensor = torch.einsum('abc, ijk, bopw, wefj -> aoeicpfk', left_layers, right_layers, mpo_1, mpo_2)
    shape = tensor.shape
    return tensor.reshape(shape[0]*shape[1]*shape[2]*shape[3], shape[4]*shape[5]*shape[6]*shape[7])


def compute_reduced_b(left_layers, right_layers, mps_1, mps_2, optimize=True):
    """Build the two-site effective right-hand-side vector from MPS overlap environments."""
    tensor = torch.einsum('ab, jk, apq, qdj -> bpdk', left_layers, right_layers, mps_1, mps_2)
    return tensor.reshape(-1)


def define_r_for_local_residual(A, b, svd_solution, r_initial, local_tol):
    """Choose an SVD rank by testing the residual of a truncated local solution."""
    u, s, vh = svd_solution
    r = max(r_initial-3, 0)
    residual = float('inf')
    while residual > local_tol and r < s.numel():
        r += 1
        x = ((u[:, :r]*s[:r]) @ vh[:r, :]).reshape(-1)
        residual = (torch.linalg.vector_norm(A @ x - b)/torch.linalg.vector_norm(b)).item()
    return max(r, 1)


def compute_pair_of_tensors(A, b, shape, direction, eps, bond_limit, r_initial, local_residual_strategy):
    """Solve the local dense linear system and split its two-site solution with truncated SVD.
    
    Parameters:
        A: Local effective system matrix.
        b: Local right-hand-side vector.
        shape: Left or right boundary bond dimension used when reshaping.
        direction: Sweep direction: left_to_right or right_to_left.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        r_initial: Current bond dimension before rank selection.
        local_residual_strategy: Optional rank-selection rule based on the local linear-system residual.
    
    Returns:
        Pair [left_core, right_core] after solving and SVD truncation."""
    solution = torch.linalg.solve(A, b)
    if direction == 'left_to_right':
        solution = solution.reshape(shape*2, -1)
    else:
        solution = solution.reshape(-1, shape*2)
    u, s, vh = torch.linalg.svd(solution, full_matrices=False)
    if not local_residual_strategy[0]:
        bond = min(max(int((s/s.sum().clamp_min(torch.finfo(s.dtype).tiny) > eps).sum().item()), 1), bond_limit)
    else:
        bond = min(define_r_for_local_residual(A, b, (u, s, vh), r_initial, local_residual_strategy[1]), bond_limit)
    if direction == 'left_to_right':
        tensor_1 = u[:, :bond].reshape(shape, 2, -1)
        tensor_2 = ((s[:bond, None]*vh[:bond, :])).reshape(bond, 2, -1)
    else:
        tensor_2 = vh[:bond, :].reshape(bond, 2, shape)
        tensor_1 = (u[:, :bond]*s[:bond]).reshape(-1, 2, bond)
    return [tensor_1, tensor_2]


def initialize_layers(mps_train, b_mps, mpo_train, optimize=True):
    """Precompute the right MPS-overlap and effective-MPO environments for the initial sweep.
    
    Parameters:
        mps_train: Value used by this routine; see the function signature and body.
        b_mps: Right-hand-side MPS in A x = b.
        mpo_train: MPO for the left-hand operator A.
        optimize: Compatibility flag retained by the local contraction API.
    
    Returns:
        Tuple (right_operator_layers, right_overlap_layers)."""
    n = len(mps_train)
    dtype, device = mpo_train[0].dtype, mpo_train[0].device
    right_mpo_layers = [torch.ones((1, 1, 1), dtype=dtype, device=device)]
    right_mps_layers = [torch.ones((1, 1), dtype=dtype, device=device)]
    for i in range(1, n-1):
        right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [torch.conj(mps_train[-i]), mpo_train[-i], mps_train[-i]], optimize))
        right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i], torch.conj(mps_train[-i])], optimize))
    return right_mpo_layers, right_mps_layers


def sweep(mps_train_, b_mps, mpo_train, right_mps_layers_, right_mpo_layers_, eps, bond_limit, optimize, local_residual_strategy):
    """Perform one left-to-right and one right-to-left two-site alternating optimisation sweep.
    
    Parameters:
        mps_train_: Current MPS estimate before this sweep.
        b_mps: Right-hand-side MPS in A x = b.
        mpo_train: MPO for the left-hand operator A.
        right_mps_layers_: Previously computed right overlap environments.
        right_mpo_layers_: Previously computed right effective-operator environments.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        optimize: Compatibility flag retained by the local contraction API.
        local_residual_strategy: Optional rank-selection rule based on the local linear-system residual.
    
    Returns:
        Tuple (updated_mps, right_overlap_layers, right_operator_layers)."""
    mps_train = mps_train_.copy()
    right_mps_layers = right_mps_layers_.copy()
    right_mpo_layers = right_mpo_layers_.copy()
    n = len(mps_train)
    dtype, device = mpo_train[0].dtype, mpo_train[0].device
    left_mpo_layers = [torch.ones((1, 1, 1), dtype=dtype, device=device)]
    left_mps_layers = [torch.ones((1, 1), dtype=dtype, device=device)]

    # Вычисление тензоров слева направо
    for i in range(n-1):
        r_initial = mps_train[i].shape[-1]
        local_A = compute_reduced_A(left_mpo_layers[i], right_mpo_layers[-1-i], mpo_train[i], mpo_train[i+1], optimize)
        local_b = compute_reduced_b(left_mps_layers[i], right_mps_layers[-1-i], b_mps[i], b_mps[i+1], optimize)
        solution = compute_pair_of_tensors(local_A, local_b, left_mps_layers[i].shape[-1], 'left_to_right', eps, bond_limit, r_initial, local_residual_strategy)
        mps_train[i] = solution[0]
        if i < n-2:
            left_mpo_layers.append(compute_mpo_layer_from_left_to_right(left_mpo_layers[-1], [torch.conj(mps_train[i]), mpo_train[i], mps_train[i]], optimize))
            left_mps_layers.append(compute_mps_layer_from_left_to_right(left_mps_layers[-1], [b_mps[i], torch.conj(mps_train[i])], optimize))
    mps_train[-1] = solution[1]

    right_mpo_layers = [torch.ones((1, 1, 1), dtype=dtype, device=device)]
    right_mps_layers = [torch.ones((1, 1), dtype=dtype, device=device)]
    # Вычисление тензоров справа налево
    for i in range(n-1):
        r_initial = mps_train[-i-2].shape[-1]
        local_A = compute_reduced_A(left_mpo_layers[-i-1], right_mpo_layers[i], mpo_train[-i-2], mpo_train[-i-1], optimize)
        local_b = compute_reduced_b(left_mps_layers[-i-1], right_mps_layers[i], b_mps[-i-2], b_mps[-i-1], optimize)
        solution = compute_pair_of_tensors(local_A, local_b, right_mps_layers[i].shape[-1], 'right_to_left', eps, bond_limit, r_initial, local_residual_strategy)
        mps_train[-i-1] = solution[1]
        if i < n-2:
            right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [torch.conj(mps_train[-i-1]), mpo_train[-i-1], mps_train[-i-1]], optimize))
            right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i-1], torch.conj(mps_train[-i-1])], optimize))
    mps_train[0] = solution[0]
    return mps_train, right_mps_layers, right_mpo_layers


def renormalize_mps(mps, initial_norm):
    """Rescale the first MPS tensor to enforce a specified global state norm."""
    mps[0] = mps[0] * (initial_norm/compute_norm(mps))


def CN_solve(mpo, mps_b, mps_initial_guess, delta_t, eps=1e-9, optimize=True, bond_limit=1000, sweep_number=1, local_residual_strategy=[False, 1e-4], renormalize=True, cn_operators=None, right_part=None):

    # Инициализация MPO и MPS из объектов
    """Approximately solve the Crank–Nicolson linear system by alternating two-site sweeps.
    
    Parameters:
        mpo: MPO tensor list with shape (left_bond, physical_out, physical_in, right_bond).
        mps_b: Right-hand-side state in the time-step equation.
        mps_initial_guess: MPS starting point for the sweep iteration.
        delta_t: Time-step size in the same units as the Hamiltonian.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        optimize: Compatibility flag retained by the local contraction API.
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        sweep_number: Number of complete left-right/right-left optimisation sweeps.
        local_residual_strategy: Optional rank-selection rule based on the local linear-system residual.
        renormalize: Whether to rescale the resulting state to the input norm.
        cn_operators: Value used by this routine; see the function signature and body.
        right_part: Precomputed right-hand-side state.
    
    Returns:
        New MPS approximating the solution of the CN linear system.
    
    Notes:
        The optional renormalize=True rescales the result after solving; disable it when diagnosing norm conservation."""
    mps_train, hamiltonian_mpo, b_mps = mps_initial_guess.copy(), mpo.copy(), mps_b.copy()
    # Ограничиваем ранг guess ДО построения локальных матриц: иначе они могут быть огромными.
    if get_bond(mps_train) > bond_limit:
        mps_train = mps_compression(mps_train, eps=0, bond_limit=bond_limit)
    # Приведение mps в правую каноническую форму
    mps_train = right_canonicalize(mps_train)
    # Первоначальная инициализация слоев сети
    mpo_ls = make_cn_operators(hamiltonian_mpo, delta_t) if cn_operators is None else cn_operators
    mps_rp = compute_cn_right_part(hamiltonian_mpo, b_mps, delta_t) if right_part is None else right_part
    right_mpo_layers, right_mps_layers = initialize_layers(mps_train, mps_rp, mpo_ls, optimize)

    # Выполнение нескольких свипов вычисления решения
    for _ in range(sweep_number):
        mps_train, right_mps_layers, right_mpo_layers = sweep(mps_train, mps_rp, mpo_ls, right_mps_layers, right_mpo_layers, eps, bond_limit, optimize, local_residual_strategy)

    if renormalize:
        renormalize_mps(mps_train, compute_norm(mps_b))
    return mps_train


def compute_discrepancy(operator, solution, right_part):
    # Точная глобальная относительная невязка текущей дискретной системы
    """Compute the relative global residual ||A x - b|| / ||b|| in tensor-train form."""
    lhs = mpo_x_mps(operator, solution)
    rhs = mps_mul_num(right_part, -1)
    return (compute_norm(mps_sum(lhs, rhs))/compute_norm(right_part)).item()


def setup_guess(initial_guess, state, operator, delta_t, bond_limit=None, h_state=None):
    """Choose the initial MPS approximation according to the configured guess strategy.
    
    Parameters:
        initial_guess: Initial-state strategy; may be a named mode or a configuration list.
        state: MPS representing the initial or current state.
        operator: MPO operator supplied to the method.
        delta_t: Time-step size in the same units as the Hamiltonian.
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        h_state: Value used by this routine; see the function signature and body.
    
    Returns:
        MPS to use as the starting point for the next solve."""
    if initial_guess == 'best':
        return state
    elif isinstance(initial_guess, list):
        guess_bond_limit = initial_guess[2] if bond_limit is None else min(initial_guess[2], bond_limit)
        if initial_guess[0] in ('approx', 'adaptive', 'local_residual_strategy'):
            return mps_compression(state, initial_guess[1], guess_bond_limit)
        elif initial_guess[0] == 'evolution_op':
            if h_state is None:
                h_state = mpo_x_mps(operator, state)
            return mps_compression(mps_sum(state, mps_mul_num(h_state, -1j*delta_t)), initial_guess[1], guess_bond_limit)
        else:
            raise ValueError('Invalid initial_guess list format')
    else:
        raise ValueError('Invalid initial_guess type')


def get_bond(mps):
    """Return the largest virtual bond dimension among MPS tensors."""
    return max(max(t.shape[0], t.shape[-1]) for t in mps)


def CN_evolute(state, operator, steps_number, delta_t, solver, initial_guess='best', eps=1e-6, chart=True, bond_limit=1000, sweep_number=1, renormalize=True, way=False, chart_every=1, right_part_eps=1e-9, right_part_bond_limit=1000):
    """Evolve the MPS through successive Crank–Nicolson time steps and record diagnostics.
    
    Parameters:
        state: MPS representing the initial or current state.
        operator: MPO operator supplied to the method.
        steps_number: Number of time steps to advance.
        delta_t: Time-step size in the same units as the Hamiltonian.
        solver: Callable that solves one time step.
        initial_guess: Initial-state strategy; may be a named mode or a configuration list.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        chart: Whether to draw diagnostic plots during/after evolution.
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        sweep_number: Number of complete left-right/right-left optimisation sweeps.
        renormalize: Whether to rescale the resulting state to the input norm.
        way: Optional output path prefix for periodically pickled state lists.
        chart_every: Value used by this routine; see the function signature and body.
        right_part_eps: Value used by this routine; see the function signature and body.
        right_part_bond_limit: Value used by this routine; see the function signature and body.
    
    Returns:
        List of MPS solutions including the initial state.
    
    Notes:
        The output stores every MPS, including the initial state; extensive trajectories may require substantial RAM."""
    list_for_matrices = [state]
    times_container = [0]
    norms = [1.0]
    bonds = [get_bond(state)]
    discrepancy_storage = [0]
    initial_norm = compute_norm(state)

    if isinstance(initial_guess, list) and initial_guess[0] == 'local_residual_strategy':
        local_residual_strategy = initial_guess[3]
    else:
        local_residual_strategy = [False, 1e-4]

    if isinstance(initial_guess, list) and initial_guess[0] in ('adaptive', 'evolution_op') and bond_limit > initial_guess[6]:
        warnings.warn('bond_limit > initial_guess[6]: адаптация ранга отключена, укажите начальный bond_limit отдельно.')

    # Инициализация оператора A = I + i*dt*H/2 один раз.
    mpo_ls = make_cn_operators(operator, delta_t)

    if chart:
        import matplotlib.pyplot as plt
        from IPython.display import display, clear_output
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        lines = [axes[0, 0].plot(range(steps_number+1), [0]*(steps_number+1))[0],
                 axes[0, 1].plot(range(steps_number+1), [0]*(steps_number+1))[0],
                 axes[1, 0].plot(range(steps_number+1), [0]*(steps_number+1))[0],
                 axes[1, 1].plot(range(steps_number+1), [0]*(steps_number+1))[0]]
        for ax, title in zip(axes.flat, ['Time', 'Bond', 'Norm', 'CN residual']):
            ax.set_title(title)
            ax.grid(alpha=0.3)

    start_time = time.time()
    i = 1
    try:
        while i < steps_number + 1:
            # Одна операция H|psi> для guess и правой части (вместо двух).
            state_current = list_for_matrices[-1]
            h_state = mpo_x_mps(operator, state_current)
            guess = setup_guess(initial_guess, state_current, operator, delta_t, bond_limit=bond_limit, h_state=h_state)
            right_part = compute_cn_right_part(operator, state_current, delta_t, eps=right_part_eps, bond_limit=right_part_bond_limit, h_state=h_state)
            solution = solver(operator, list_for_matrices[-1], guess, delta_t, eps=eps, bond_limit=bond_limit, sweep_number=sweep_number, local_residual_strategy=local_residual_strategy, renormalize=renormalize, cn_operators=mpo_ls, right_part=right_part)
            discrepancy = compute_discrepancy(mpo_ls, solution, right_part)

            # При rollback=0 текущий шаг принимаем; увеличенный bond_limit используем на следующем шаге.
            if isinstance(initial_guess, list) and initial_guess[0] in ('adaptive', 'evolution_op') and discrepancy > initial_guess[4] and bond_limit < initial_guess[6]:
                rollback = min(initial_guess[5], len(list_for_matrices)-1)
                bond_limit = min(bond_limit+initial_guess[3], initial_guess[6])
                if rollback > 0:
                    del list_for_matrices[-rollback:]
                    del times_container[-rollback:]
                    del norms[-rollback:]
                    del bonds[-rollback:]
                    del discrepancy_storage[-rollback:]
                    i = len(list_for_matrices)
                    continue

            list_for_matrices.append(solution)
            times_container.append(time.time()-start_time)
            norms.append(round((compute_norm(solution)/initial_norm).item(), 3))
            bonds.append(get_bond(solution))
            discrepancy_storage.append(round(discrepancy, 4))

            if chart and (i % chart_every == 0 or i == steps_number):
                for line, values in zip(lines, [times_container, bonds, norms, discrepancy_storage]):
                    line.set_ydata(values + [0]*(steps_number+1-len(values)))
                for ax in axes.flat:
                    ax.relim()
                    ax.autoscale_view()
                clear_output(wait=True)
                display(fig)

            if i % 10 == 0 and isinstance(way, str) and way:
                with open(way+'solutions.pkl', 'wb') as f:
                    pickle.dump(list_for_matrices, f)
            i += 1
    except KeyboardInterrupt:
        print('Цикл остановлен пользователем.')

    return list_for_matrices
