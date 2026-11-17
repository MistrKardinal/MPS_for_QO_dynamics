import torch
import math
import time
import pickle
import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import add

def vec_to_mps(psi_vec, eps = 1e-6):

    """Decompose a length-2**n state vector into binary-physical MPS tensors via SVD.
    
    Parameters:
        psi_vec: Value used by this routine; see the function signature and body.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
    
    Returns:
        List of rank-three MPS cores (one core per binary site)."""
    out_dtype = psi_vec.dtype

    if not torch.is_tensor(psi_vec):
        raise TypeError("Аргумент должен быть массивом NumPy (torch.array)!")
    if not math.log2(len(psi_vec)).is_integer():
        raise ValueError("Длина входного вектора должна быть степенью двойки")
    tensors = []
    
    n = int(math.log2(len(psi_vec)))
    shape = (2, 2**(n-1))
    psi_matrix = psi_vec.reshape(shape)

    for _ in range(n-1):
        psi_svd = torch.linalg.svd(psi_matrix, full_matrices = False)
        bond = int((psi_svd.S / psi_svd.S.sum() > eps).sum())

        psi_svd_U = psi_svd.U.reshape(int(psi_svd.U.shape[0]/2), 2, int(psi_svd.U.shape[1]))[:, :, :bond]
        tensors.append(psi_svd_U.to(out_dtype))

        pmatrix = torch.diag(psi_svd.S.to(out_dtype)[:bond]) @ psi_svd.Vh.to(out_dtype)[:bond, :]
        psi_matrix = pmatrix.reshape(int(pmatrix.shape[0]*2), int(pmatrix.shape[1]/2))

    last_tensor = pmatrix.reshape(int(pmatrix.shape[0]), 2, -1)
    tensors.append(last_tensor)
    return tensors


def mpo_multiplication(mpo_one, mpo_two):
    """Multiply two MPOs by contracting matching local input/output indices."""
    mult_mpo = []
    for i in range(len(mpo_one)):
        sh1 = mpo_one[i].shape
        sh2 = mpo_two[i].shape
        mult_mpo.append(torch.einsum('ijkl, mkop -> imjolp', mpo_one[i], mpo_two[i]).reshape(sh1[0]*sh2[0], 2, 2, sh1[3]*sh2[3]))
    return mult_mpo


def x_mpo(n, x0 = 0, step = 1, dtype=torch.complex128):
    """Represent the discrete diagonal coordinate operator for a uniform 2**n-point grid."""
    x_mpo_first = [torch.tensor([1, 0, 0, 0, 0, 0, 1, 1], dtype=dtype).reshape(1, 2, 2, 2)]
    x_mpo_intermediate = [torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2], dtype=dtype).reshape(2, 2, 2, 2)]
    x_mpo_last = [torch.tensor([x0/step, 0, 0, x0/step + 1, 2, 0, 0, 2], dtype=dtype).reshape(2, 2, 2, 1) * step]
    return x_mpo_first + (n - 2) * x_mpo_intermediate + x_mpo_last


def x2_mpo(n, x0=0, step=1, dtype=torch.complex128):
    """Construct a compact MPO for the squared coordinate operator on the same grid."""
    A = torch.tensor([[1,0],[0,1]], dtype=dtype)
    B = torch.tensor([[0,0],[0,1]], dtype=dtype)
    C = torch.zeros(2,2, dtype=dtype)
    M_1 = torch.stack([A, B, B], dim=-1).reshape(1,2,2,3).to(dtype)
    block1 = torch.stack([A, B, B], dim=-1)
    block2 = torch.stack([C, 2*A, 4*B], dim=-1)
    block3 = torch.stack([C, C, 4*A], dim=-1)
    M_2 = torch.stack([block1, block2, block3], dim=0).to(dtype)
    M_3 = torch.stack([(B+A*x0/step)**2, 4*(B+A*x0/step), 4*A], dim=0).reshape(3,2,2,1).to(dtype)

    return [M_1]+[M_2]*(n-2)+[M_3*step**2]


def der_mpo(n, step = 1, dtype=torch.complex128):
    """Build the provided discrete first-derivative stencil as an MPO."""
    der_mpo_list = []
    mpo_derivative_first = 0.5 * torch.tensor([1,0,0,0,1,0,0,0,-1,1,0,0], dtype=dtype).reshape(1,2,2,3) / step
    mpo_derivative_last = torch.tensor([0,1,-1,0,0,0,1,0,0,1,0,0], dtype=dtype).reshape(3,2,2,1)
    mpo_derivative_intermediate = torch.tensor([1,0,0,0,1,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], dtype=dtype).reshape(3,2,2,3)
    der_mpo_list.append(mpo_derivative_first)
    for i in range(n-2):
        der_mpo_list.append(mpo_derivative_intermediate)
    der_mpo_list.append(mpo_derivative_last)
    return der_mpo_list


def der2_mpo(n, step = 1, dtype=torch.complex128):
    """Build the provided discrete second-derivative finite-difference stencil as an MPO."""
    mpo_derivative2_first = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=dtype).reshape(1, 2, 2, 3) / step**2
    mpo_derivative2_last = torch.tensor([-2, 1, 1, -2, 0, 0, 1, 0, 0, 1, 0, 0], dtype=dtype).reshape(3, 2, 2, 1)
    mpo_derivative2_intermediate = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0], dtype=dtype).reshape(3, 2, 2, 3)
    return [mpo_derivative2_first] + [mpo_derivative2_intermediate] * (n - 2) + [mpo_derivative2_last]


def one_mpo(n, dtype=torch.complex128):
    """Construct an identity MPO for n binary physical sites."""
    massive = [torch.eye(2, dtype=dtype).reshape(1,2,2,1)] * n
    return massive


def check_canonical_form(mps, tol=1e-6):

    """Inspect left- and right-orthonormality of each MPS core against a tolerance."""
    length = len(mps)

    def is_close_to_identity(matrix, tol):

        """Test whether a square matrix is close to the identity in Frobenius norm."""
        n = matrix.size(0)
        identity = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        
        diff_norm = torch.norm(matrix - identity, p='fro')
        
        return diff_norm < tol


    condition = ['']*length
    lc_condition_check = ['left_n']*(length-1) + ['unnorm']
    rc_condition_check = ['unnorm'] + ['right_n']*(length-1)

    for i, tensor in enumerate(mps):
        rc_condition = torch.einsum("abc, dbc -> ad", tensor, torch.conj(tensor))
        lc_condition = torch.einsum("abc, abd -> cd", torch.conj(tensor), tensor)
        if is_close_to_identity(rc_condition, tol):
            condition[i] += 'right_n'
        if is_close_to_identity(lc_condition, tol):
            condition[i] += 'left_n'
        if not is_close_to_identity(rc_condition, tol) and not is_close_to_identity(lc_condition, tol):
            condition[i] += 'unnorm'

    return condition


def tensors_multiplication(mps):

    """Contract all MPS cores to reconstruct the full state vector."""
    if not isinstance(mps, list):
        raise TypeError("Аргумент должен быть списком")

    tensor_second = mps[0]
    for i in range(len(mps)-1):
        tensor_second = torch.einsum("abc, cde -> abde", tensor_second, mps[i+1])
        shape = tensor_second.shape
        tensor_second = tensor_second.reshape(shape[0], shape[1]*shape[2], shape[3])
    return tensor_second.reshape(-1)


def QR_rc_decomposition(psi):

    #if not isinstance(psi, torch.ndarray):
    #    raise TypeError("Аргумент должен быть массивом NumPy (torch.array)!")
    #if not torch.log2(len(psi)).is_integer():
    #    raise ValueError("Длина входного вектора должна быть степенью двойки")

    """Convert a dense state vector into a right-canonical MPS using QR factorisation."""
    n = int(torch.log2(torch.tensor(len(psi))))
    arr_store = []
    shape = (2**(n-1), 2)
    psi_matrix = psi.reshape(shape)
    for i in range(0,n-1):
        Q, R = torch.linalg.qr(psi_matrix.T, mode = 'reduced')
        Q_reshaped = Q.T.reshape(int(Q.T.shape[0]),2,int(Q.T.shape[1]/2)) # Переводим матрицу Q в тензор
        arr_store.append(Q_reshaped)
        pmatrix = R.T 
        psi_matrix = pmatrix.reshape(int(pmatrix.shape[0]/2),int(pmatrix.shape[1]*2))    
    Q = pmatrix.reshape(int(pmatrix.shape[0]/2),2,int(pmatrix.shape[1]))
    arr_store.append(Q)
    return arr_store[::-1]


def mpo_sum(mpo_one, mpo_two):
    """Construct the sum of two MPOs by combining their virtual-bond spaces."""
    common_tensor = []
    common_tensor.append(torch.concatenate([mpo_one[0], mpo_two[0]], axis = 3).reshape(1, 2, 2, -1))
    for i in range(1,len(mpo_one)-1):
        zero_arr1 = torch.zeros((mpo_two[i].shape[0], 2, 2, mpo_one[i].shape[3]))
        zero_arr2 = torch.zeros((mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3]))
        ff1 = torch.concatenate([mpo_one[i], zero_arr1], axis=0).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_one[i].shape[3])
        ff2 = torch.concatenate([zero_arr2, mpo_two[i]], axis=0).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3])
        new_element = torch.concatenate([ff1, ff2], axis = 3).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3] + mpo_one[i].shape[3])
        common_tensor.append(new_element)
    common_tensor.append(torch.concatenate([mpo_one[-1], mpo_two[-1]], axis = 0).reshape(mpo_one[-1].shape[0] + mpo_two[-1].shape[0], 2, 2, 1))
    return common_tensor


def mpo_mul_num(mpo, num):
    """Multiply an MPO by a scalar by rescaling its first tensor only."""
    mpo_ch = mpo.copy()
    mpo_ch[0] = mpo_ch[0] * num
    return mpo_ch


def chislo(n, k, dtype=torch.complex128):
    """Construct k times an identity MPO on n binary sites."""
    massive = [torch.eye(2, dtype=dtype).reshape(1,2,2,1)] * n
    massive[0] = massive[0] * k
    return massive


def mpo_to_matrix(mpo):
    """Contract an MPO into its full dense matrix (suitable only for small systems)."""
    mul = mpo[0]
    for i in range(1, len(mpo)):
        mul = torch.einsum("ijkl, lghp -> ijgkhp", mul, mpo[i])
        mul = mul.reshape(mul.shape[0], mul.shape[1]*mul.shape[2], mul.shape[3]*mul.shape[4], mul.shape[5])
    mul = mul.reshape(mul.shape[1], mul.shape[2])

    return mul


def asym_gauss_mps(x_left, x_right, n, sigma, eps=1e-6, dtype=torch.complex128):
    """Sample a Gaussian wavefunction on a non-symmetric x grid and encode it as an MPS."""
    x_values = torch.linspace(-x_left, x_right, 2**n, dtype=dtype)
    x_squared = x_values**2
    gauss_state = torch.exp(-x_squared / (2 * sigma**2)) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
    return vec_to_mps(gauss_state, eps=eps)


def mps_compression(mps, eps = 1e-6, bond_limit = 1000):

    """Reduce the virtual-bond dimensions using truncated singular-value decompositions.
    
    Parameters:
        mps: MPS tensor list with shape (left_bond, physical, right_bond).
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
    
    Returns:
        New list of compressed MPS cores."""
    dtype = mps[0].dtype

    if not isinstance(mps, list):
        raise TypeError("Аргумент должен быть списком")

    compressed_mps = []

    mps = left_canonical_MPS(mps)
    temp_tensor = [mps[-1]]
    for i in range(len(mps)-1):
        shapi = temp_tensor[0].shape
        element = temp_tensor[-1].reshape(shapi[0], 2 * shapi[2]) # Переводим тензор в матрицу
        element_svd = torch.linalg.svd(element, full_matrices=False) # Делаем сингулярное разложение матрицы
        b_tensor = element_svd.Vh.reshape(element_svd.Vh.shape[0], 2, shapi[2]).to(dtype) # Переводим матрицу Vh в тензор
        bond = min((element_svd.S / element_svd.S.sum() > eps).sum(), bond_limit)
        b_tensor = b_tensor[:bond, :, :] # Обрезаем тензор
        compressed_mps.append(b_tensor) # Вносим тензор в лист хранения тензоров
        element_svd_S = element_svd.S[:bond] # Обрезаем матрицу S
        newpmatrx = element_svd.U[:,:bond].to(dtype) @ torch.diag(element_svd_S.to(dtype)) # Обрезаем матрицу U и умножаем на матрицу S
        pmatrix = torch.einsum("ijk, kl -> ijl", mps[-2-i], newpmatrx) # Перемножаем более левый тензор с перемноженными матрицами U и S
        temp_tensor = [pmatrix]
    compressed_mps.append(pmatrix) # Добавляем последний тензор в лист хранения тензоров
    compressed_mps[::] = compressed_mps[::-1] # Инвертируем порядок хранения листа хранения
    return compressed_mps


def euler_solve(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps=1e-9, bond_limit=1000, sweep_number=1, renormalize=True, gpu=[torch.cuda.device_count()-1, 2000], local_residual_strategy=(False, 1e-4)):
    # The implicit Euler solver is implemented in a separate module.
    """Compatibility wrapper calling the self-contained implicit Euler solver module.
    
    Parameters:
        ex_mpo: Preassembled Euler operator in MPO form.
        ex_mps_right_part: Right-hand-side MPS of the Euler linear system.
        ex_mps_initial_guess: Initial MPS guess for the local sweeps.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        sweep_number: Number of complete left-right/right-left optimisation sweeps.
        renormalize: Whether to rescale the resulting state to the input norm.
        gpu: Legacy GPU selector; refer to this function body for device handling.
        local_residual_strategy: Optional rank-selection rule based on the local linear-system residual.
    
    Returns:
        New MPS approximating the solution of the Euler linear system.
    
    Notes:
        The imported ImplicitEuler_torch module must be available when this wrapper is used. A solver itself remains self-contained."""
    from ImplicitEuler_torch import euler_solve as _euler_solve
    return _euler_solve(ex_mpo, ex_mps_right_part, ex_mps_initial_guess,
                        eps=eps, bond_limit=bond_limit, sweep_number=sweep_number,
                        renormalize=renormalize, gpu=gpu,
                        local_residual_strategy=local_residual_strategy)


def right_canonicalize(mps):
    """Move the MPS orthogonality centre to the left via QR decompositions from right to left."""
    rc_mps = mps.copy()
    for i in range(1, len(mps)):
        shape = rc_mps[-i].shape
        matrix = torch.conj(rc_mps[-i].reshape(shape[0], shape[1]*shape[2])).T
        q, r = torch.linalg.qr(matrix, mode = 'reduced')
        rc_mps[-i] = torch.conj(q).T.reshape(q.T.shape[0], 2, int(q.T.shape[1]/2))
        r_conj = torch.conj(r).T 
        rc_mps[-i-1] = torch.einsum("ijk, kl -> ijl", mps[-i-1], r_conj)
    
    return rc_mps


def initialize_data(train_mps, right_part_mps, mpo, dtype):
    """Cast the guess MPS, right-hand side MPS, and MPO to a common torch dtype."""
    train_mps = [tensor.to(dtype) for tensor in train_mps]
    right_part_mps = [tensor.to(dtype) for tensor in right_part_mps]
    mpo = [tensor.to(dtype) for tensor in mpo]

    return train_mps, right_part_mps, mpo

 
def initialize_layers(mps_train, b_mps, mpo_train):
    
    """Precompute the right MPS-overlap and effective-MPO environments for the initial sweep.
    
    Parameters:
        mps_train: Value used by this routine; see the function signature and body.
        b_mps: Right-hand-side MPS in A x = b.
        mpo_train: MPO for the left-hand operator A.
    
    Returns:
        Tuple (right_operator_layers, right_overlap_layers)."""
    n = len(mps_train)
    
    right_mpo_layers = [torch.ones((1,1,1), dtype=mps_train[0].dtype)]
    right_mps_layers = [torch.ones((1,1), dtype=mps_train[0].dtype)]
    
    for i in range(1, n-1):
        right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [torch.conj(mps_train[-i]), mpo_train[-i], mps_train[-i]]))    
        right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i], torch.conj(mps_train[-i])]))
        
    return right_mpo_layers, right_mps_layers


def sweep(mps_train_, b_mps, mpo_train, right_mps_layers_, right_mpo_layers_, eps, bond_limit, gpu):

    """Perform one left-to-right and one right-to-left two-site alternating optimisation sweep.
    
    Parameters:
        mps_train_: Current MPS estimate before this sweep.
        b_mps: Right-hand-side MPS in A x = b.
        mpo_train: MPO for the left-hand operator A.
        right_mps_layers_: Previously computed right overlap environments.
        right_mpo_layers_: Previously computed right effective-operator environments.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        gpu: Legacy GPU selector; refer to this function body for device handling.
    
    Returns:
        Tuple (updated_mps, right_overlap_layers, right_operator_layers)."""
    mps_train = mps_train_.copy()
    right_mps_layers = right_mps_layers_.copy()
    right_mpo_layers = right_mpo_layers_.copy()

    n = len(mps_train)

    left_mpo_layers = [torch.ones((1,1,1), dtype=mps_train[0].dtype)]
    left_mps_layers = [torch.ones((1,1), dtype=mps_train[0].dtype)]

    # Вычисление тензоров слева направо
    for i in range(n-1):
        local_A = compute_reduced_A(left_mpo_layers[i], right_mpo_layers[-1-i], mpo_train[i], mpo_train[i+1])
        local_b = compute_reduced_b(left_mps_layers[i], right_mps_layers[-1-i], b_mps[i], b_mps[i+1])
        solution = compute_pair_of_tensors(local_A, local_b, left_mps_layers[i].shape[-1], "left_to_right", eps, bond_limit, gpu)
        mps_train[i] = solution[0]
        if i < n-2:
            left_mpo_layers.append(compute_mpo_layer_from_left_to_right(left_mpo_layers[-1], [torch.conj(mps_train[i]), mpo_train[i], mps_train[i]]))
            left_mps_layers.append(compute_mps_layer_from_left_to_right(left_mps_layers[-1], [b_mps[i], torch.conj(mps_train[i])]))
    mps_train[-1] = solution[1]

    right_mpo_layers = [torch.ones((1,1,1), dtype=mps_train[0].dtype)]
    right_mps_layers = [torch.ones((1,1), dtype=mps_train[0].dtype)]

    # Вычисление тензоров справа налево
    for i in range(n-1):
        local_A = compute_reduced_A(left_mpo_layers[-i-1], right_mpo_layers[i], mpo_train[-i-2], mpo_train[-i-1])
        local_b = compute_reduced_b(left_mps_layers[-i-1], right_mps_layers[i], b_mps[-i-2], b_mps[-i-1])
        solution = compute_pair_of_tensors(local_A, local_b, right_mps_layers[i].shape[-1], "right_to_left", eps, bond_limit, gpu)
        mps_train[-i-1] = solution[1]
        if i < n-2:
            right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [torch.conj(mps_train[-i-1]), mpo_train[-i-1], mps_train[-i-1]]))
            right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i-1], torch.conj(mps_train[-i-1])]))
    mps_train[0] = solution[0]

    return mps_train, right_mps_layers, right_mpo_layers   


def compute_mps_layer_from_left_to_right(layer, tensors):
    """Extend a left boundary environment for an MPS overlap by one physical site."""
    new_layer = torch.einsum('ab, aij -> jib', layer, tensors[0])
    new_layer = torch.einsum('abc, cbj -> aj', new_layer, tensors[1])
    return new_layer


def compute_mps_layer_from_right_to_left(layer, tensors):
    """Extend a right boundary environment for an MPS overlap by one physical site."""
    new_layer = torch.einsum('ab, ija -> ijb', layer, tensors[0])
    new_layer = torch.einsum('abc, ibc -> ai', new_layer, tensors[1])
    return new_layer


def compute_mpo_layer_from_left_to_right(layer, tensors):
    """Update the left effective-operator environment by contracting one MPO/MPS site."""
    new_layer = torch.einsum('abc, ajk -> kjbc', layer, tensors[0])
    new_layer = torch.einsum('abcd, cbij -> ajid', new_layer, tensors[1])
    new_layer = torch.einsum('abcd, dci -> abi', new_layer, tensors[2])
    return new_layer


def compute_mpo_layer_from_right_to_left(layer, tensors):
    """Update the right effective-operator environment by contracting one MPO/MPS site."""
    new_layer = torch.einsum('abc, ija -> ijbc', layer, tensors[0])
    new_layer = torch.einsum('abcd, ibjc -> aijd', new_layer, tensors[1])
    new_layer = torch.einsum('abcd, icd -> abi', new_layer, tensors[2])
    return new_layer


def compute_reduced_A(left_layers, right_layers, mpo_1, mpo_2):
    """Build the two-site effective matrix of the linear system from environments and two MPO cores."""
    tensor = torch.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers, right_layers, mpo_1, mpo_2)
    shape = tensor.shape
    return tensor.reshape(shape[0]*shape[1]*shape[2]*shape[3], shape[4]*shape[5]*shape[6]*shape[7])


def compute_reduced_b(left_layers, right_layers, mps_1, mps_2):
    """Build the two-site effective right-hand-side vector from MPS overlap environments."""
    tensor = torch.einsum("ab, jk, apq ,qdj -> bpdk", left_layers, right_layers, mps_1, mps_2)
    return tensor.reshape(-1)


def compute_pair_of_tensors(A, b, shape, direction, eps, bond_limit, gpu):

    """Solve the local dense linear system and split its two-site solution with truncated SVD.
    
    Parameters:
        A: Local effective system matrix.
        b: Local right-hand-side vector.
        shape: Left or right boundary bond dimension used when reshaping.
        direction: Sweep direction: left_to_right or right_to_left.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        gpu: Legacy GPU selector; refer to this function body for device handling.
    
    Returns:
        Pair [left_core, right_core] after solving and SVD truncation."""
    if torch.cuda.is_available() and len(b) > gpu[1] and gpu[1] is not None:
        torch.cuda.set_device(gpu[0])
        A = A.cuda()
        b = b.cuda()
        solution = torch.linalg.solve(A, b).to("cpu")
        print('g', end='')
    else:
        solution = torch.linalg.solve(A, b)

    if direction == "left_to_right":
        solution = solution.reshape(shape*2, -1)
    else:
        solution = solution.reshape(-1, shape*2)

    svd_solution = torch.linalg.svd(solution, full_matrices = False)
    bond = min((svd_solution.S / svd_solution.S.sum() > eps ).sum(), bond_limit)

    if direction == "left_to_right":
        tensor_1 = svd_solution.U.to(A.dtype).reshape(shape, 2, -1)[:, :, :bond]
        tensor_2 = (torch.diag(svd_solution.S.to(A.dtype)[:bond]) @ svd_solution.Vh.to(A.dtype)[:bond, :]).reshape(tensor_1.shape[-1], 2, -1)
    else:
        tensor_2 = svd_solution.Vh.to(A.dtype).reshape(-1, 2, shape)[:bond, :, :]
        tensor_1 = (svd_solution.U.to(A.dtype)[:, :bond] @ torch.diag(svd_solution.S.to(A.dtype)[:bond])).reshape(-1, 2, tensor_2.shape[0])[:, :, :bond]
    return [tensor_1, tensor_2]


def compute_norm(mps):
    """Calculate the Euclidean norm of an MPS without reconstructing its full vector."""
    mps_for_change = mps.copy()
    mps_for_change = left_canonical_MPS(mps_for_change)
    return calculate_lc_mpsnorm(mps_for_change)


def renormalize_mps(mps, initial_norm):
    """Rescale the first MPS tensor to enforce a specified global state norm."""
    mps[0] = mps[0] * initial_norm / calculate_rc_mpsnorm(mps)


def calculate_lc_mpsnorm(mps):
    """Compute the state norm from the last tensor of a left-canonical MPS."""
    return (torch.sqrt(torch.trace(torch.conj(mps[-1])[:, 0, :].T @ mps[-1][:, 0, :]) + torch.trace(torch.conj(mps[-1])[:, 1, :].T @ mps[-1][:, 1, :]))).abs()


def calculate_rc_mpsnorm(mps):
    """Compute the state norm from the first tensor of a right-canonical MPS."""
    return (torch.sqrt(torch.trace(torch.conj(mps[0])[:, 0, :].T @ mps[0][:, 0, :]) + torch.trace(torch.conj(mps[0])[:, 1, :].T @ mps[0][:, 1, :]))).abs()


def left_canonical_MPS(mps):

    """Transform the tensor train to a left-canonical MPS by sequential QR factorizations."""
    if not isinstance(mps, list):
        raise TypeError("Аргументы должны быть списками")

    mps_copy = mps.copy()
    left_canonical_mps = [] # Создаём лист для хранения тензоров
    temporary_element = [mps_copy[0]]
    for i in range(len(mps_copy)-1):
        sh = temporary_element[0].shape
        element = temporary_element[0].reshape(sh[0] * 2, sh[2]) # Переводим тензор в матрицу
        element_q, element_r = torch.linalg.qr(element) # Делаем сингулярное разложение матрицы
        tensor = element_q.reshape(sh[0], 2, -1) # Переводим матрицу Vh в тензор
        left_canonical_mps.append(tensor) # Вносим тензор в лист хранения тензоров   
        newpmatrx = element_r
        pmatrix = torch.einsum("ab, bjk -> ajk", newpmatrx, mps_copy[i+1]) # Перемножаем более левый тензор с перемноженными матрицами U и S
        temporary_element = [pmatrix]
    left_canonical_mps.append(pmatrix) # Добавляем последний тензор в лист хранения тензоров
    return left_canonical_mps


def mps_sum(psi_one, psi_two):

    """Represent the sum of two MPS states through a direct sum of virtual bonds."""
    if not isinstance(psi_one, list) or not isinstance(psi_two, list):
        raise TypeError("Аргументы должны быть списками")
    if len(psi_one) != len(psi_two):
        raise ValueError("Длины двух mps должны быть одинаковы")

    sum_mps = []
    sum_mps.append(torch.concatenate([psi_one[0], psi_two[0]], axis = 2).reshape(1, 2, -1))
    for i in range(1, len(psi_one)-1):
        zero_arr1 = torch.zeros((psi_two[i].shape[0], 2, psi_one[i].shape[2]))
        zero_arr2 = torch.zeros((psi_one[i].shape[0], 2, psi_two[i].shape[2]))
        ff1 = torch.concatenate([psi_one[i], zero_arr1], axis=0).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_one[i].shape[2])
        ff2 = torch.concatenate([zero_arr2, psi_two[i]], axis=0).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_two[i].shape[2])
        new_element = torch.concatenate([ff1, ff2], axis = 2).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_two[i].shape[2] + psi_one[i].shape[2])
        sum_mps.append(new_element)
    sum_mps.append(torch.concatenate([psi_one[-1], psi_two[-1]], axis = 0).reshape(psi_one[-1].shape[0] + psi_two[-1].shape[0], 2, 1))
    return sum_mps


def get_bond(mps):
    """Return the largest virtual bond dimension among MPS tensors."""
    dim1 = []
    dim2 = []
    for i in mps:
        dim1.append(i.shape[0])
        dim2.append(i.shape[-1])
    return max(max(dim1), max(dim2)) 


def mps_mul_num(mps, num):
    """Multiply an MPS by a scalar by rescaling its first tensor only."""
    mps_in_change = mps.copy()
    mps_in_change[0] = mps_in_change[0] * num
    return mps_in_change


def mpo_x_mps(mpo, mps):
    """Apply an MPO to an MPS, without automatically compressing the resulting bonds."""
    new_mps = []
    for i in range(len(mps)):
        new_mps.append(torch.einsum("ijnl, mno -> imjlo", mpo[i], mps[i]).reshape(mpo[i].shape[0] * mps[i].shape[0], 2, mpo[i].shape[-1] * mps[i].shape[-1]))
    return new_mps


def evolute_TDVP(state, operator, steps_number, delta_t, solver, initial_guess="best", eps=1e-6, chart=True, bond_limit=1000, sweep_number=1, renormalize=True, way=False):
    """Run a supplied time-stepping solver and monitor states, errors and bond dimensions."""
    list_for_matrices = [state]
    times_container = []
    norms = []
    bonds = []

    start_time = time.time()

    if chart == True:
        fig, axes, lines, texts = setup_plot(steps_number)
    
    i = 1
    try:
        while i < steps_number + 1:
            solution = solver(list_for_matrices[-1], operator, delta_t, eps=eps, bond_limit=bond_limit)
            discrepancy = compute_discrepancy(operator, solution, list_for_matrices[-1], delta_t)
            
            i = update_iteration(i, initial_guess, bond_limit, discrepancy)
            list_for_matrices, times_container, norms, bonds = update_lists(list_for_matrices, times_container, norms, bonds, solution, start_time, initial_guess, bond_limit, discrepancy)
            bond_limit = update_bond_limit(initial_guess, bond_limit, discrepancy)
            if chart == True and i > 0:
                update_plot(i, times_container, bonds, norms, steps_number, lines, axes, texts, fig, bond_limit, np.round(discrepancy, 3), cut_time_base=10)

            if i != 0 and i % 10 == 0 and isinstance(way, str):
                with open(way+'solutions.pkl', 'wb') as f:
                    pickle.dump(list_for_matrices, f)
            i += 1

    except KeyboardInterrupt:
        print("Цикл остановлен пользователем.")
    
    return list_for_matrices, times_container, norms, bonds

    
def calculate_photon_number_in_set_of_solutions(set_of_solutions, n1, n2, x_bond1, x_bond2, step1, step2, x_bias, solver, bound = False, step = 1, chart = True):
    """Evaluate pump and signal photon-number observables at selected saved time steps."""
    if chart == True:
        real_length = len(range(1, len(set_of_solutions)+1, step))
        fig, ax = plt.subplots()
        line1, = ax.plot(torch.arange(1, real_length + 1), torch.zeros(real_length), 'r.', label='Times')
        line2, = ax.plot(torch.arange(1, real_length + 1), torch.zeros(real_length), label='Poly Fit')
        text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center')
        ax.legend()

    times_container = []
    start_time = time.time()
    j = 0

    photon_number_in_mode1 = []
    photon_number_in_mode2 = []
    photon_number_common = []
    for i in tqdm(range(1, len(set_of_solutions)+1, step)):
        j += 1
        photon_number_in_mode1.append(torch.round(solver(set_of_solutions[i-1], 0, n1, x_bias-x_bond1, step1), 3))
        photon_number_in_mode2.append(torch.round(solver(set_of_solutions[i-1], 1, n2, -x_bond2, step2), 3))
        photon_number_common = list(map(add, photon_number_in_mode1, photon_number_in_mode2))

        times_container.append(time.time() - start_time)
        if chart == True:
            cut_time = 7
            if j < cut_time+1:
                poly_coeffs = torch.polyfit(torch.arange(1, j + 1), times_container, 2)
            else:
                poly_coeffs = torch.polyfit(torch.arange(j-(cut_time-1), j+1), times_container[j-cut_time:j], 2)
            
            poly_func = torch.poly1d(poly_coeffs)  
            line1.set_ydata(times_container + [0]*(real_length-len(times_container)))
            line2.set_ydata(poly_func(torch.arange(1, real_length + 1)))
            ax.relim()
            ax.autoscale_view()
            text.set_text(f'Общее время выполнения {torch.round(poly_func(real_length))} секунды, \n Оставшееся время {torch.round(poly_func(real_length)-times_container[j-1])} секунды')
            clear_output(wait=True)
            display(fig)
            plt.pause(0.001)  # Пауза для обновления графика
            plt.show()

    return photon_number_in_mode1, photon_number_in_mode2, photon_number_common


def calculate_photon_number3(mps, position, length, x0, step, dtype=torch.complex128):
    """Compute a mode-local quadrature oscillator observable using an MPO and MPS contractions.
    
    Parameters:
        mps: MPS tensor list with shape (left_bond, physical, right_bond).
        position: Index of the selected physical mode.
        length: Number of binary sites encoding a mode.
        x0: Coordinate of the left end of the discretized grid.
        step: Coordinate grid spacing or iteration stride (see context).
        dtype: torch dtype used when constructing the returned tensors.
    
    Returns:
        One-element tensor containing the grid-discretized oscillator observable.
    
    Notes:
        This is <N + 1/2> on the chosen discretized grid, not <N>: subtract 0.5 to report mean photons. The finite-difference derivative also introduces discretization error."""
    l = len(mps)
    photon_number_op = photon_number_operator_mpo(length, x0, step, dtype=dtype)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[-1], torch.conj(env_part[-1]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], torch.conj(env_part[-i-1]))
    else:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[0], torch.conj(env_part[0]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], torch.conj(env_part[i]))

    # Свертка матрицы плотности состояния с оператором числа фотонов
    if position == 0:
        first_state_step = torch.einsum("abc, ibjk, ejf -> aieckf", state_part[0], photon_number_op[0],  torch.conj(state_part[0]))
        for i in range(1, len(state_part)):
            first_state_step = torch.einsum("abcdef, djk, ejop, fow -> abckpw", first_state_step, state_part[i], photon_number_op[i], torch.conj(state_part[i]))
    else:
        first_state_step = torch.einsum("abc, ibjk, ejf -> aieckf", state_part[-1], photon_number_op[-1],  torch.conj(state_part[-1]))
        for i in range(1, len(state_part)):
            first_state_step = torch.einsum("abcdef, gja, hjob, woc -> ghwdef", first_state_step, state_part[-i-1], photon_number_op[-i-1], torch.conj(state_part[-i-1]))

    # Вычисление следа матрицы плотности
    if position == 0:
        rho_trace_first_step = torch.einsum("abc, ibk -> aick", state_part[0], torch.conj(state_part[0]))
        for i in range(1, len(state_part)):
            rho_trace_first_step = torch.einsum("abcd, cij, dif -> abjf", rho_trace_first_step, state_part[i], torch.conj(state_part[i]))
    else:
        rho_trace_first_step = torch.einsum("abc, ibk -> aick", state_part[-1], torch.conj(state_part[-1]))
        for i in range(1, len(state_part)):
            rho_trace_first_step = torch.einsum("abcd, ika, jkb -> ijcd", rho_trace_first_step, state_part[-i-1], torch.conj(state_part[-i-1]))
    
    if position == 1:
        common_trace = torch.einsum("abcd, cdij -> abij", first_env_step, rho_trace_first_step).reshape(-1)
        return torch.einsum("abcd, cfdklm -> afbklm", first_env_step, first_state_step).reshape(-1)/common_trace
    else:
        common_trace = torch.einsum("abcd, cdij -> abij", rho_trace_first_step, first_env_step).reshape(-1)
        return torch.einsum("abcd, ijkahb -> ijkchd", first_env_step, first_state_step).reshape(-1)/common_trace


def photon_number_operator_mpo(n, x0, step, dtype=torch.complex128):
    """Construct the quadrature-space operator (x**2 - d**2/dx**2) / 2.
    
    Parameters:
        n: Number of binary physical sites.
        x0: Coordinate of the left end of the discretized grid.
        step: Coordinate grid spacing or iteration stride (see context).
        dtype: torch dtype used when constructing the returned tensors.
    
    Returns:
        MPO tensor list for the oscillator observable.
    
    Notes:
        For [x, p]=i, physical photon number is N=(x**2+p**2-1)/2; this helper omits the -1/2 identity term."""
    return mpo_mul_num(mpo_sum(mpo_multiplication(x_mpo(n, x0, step, dtype), x_mpo(n, x0, step, dtype)), mpo_mul_num(der2_mpo(n, step, dtype), -1)), 1/2)


def calculate_number_of_elements(mpo):
    """Count the stored scalar entries in all MPO cores."""
    count = 0
    for i in mpo:
        count += i.shape[0]*i.shape[1]*i.shape[2]*i.shape[3]
    return count


def choose_device(vec, gpu_index=torch.cuda.device_count()-1):
    """Choose CUDA for sufficiently large vectors when a CUDA device is available."""
    if torch.cuda.is_available() and len(vec) > 2000:
        device = 'cuda'
    else: 
        device = 'cpu'
    return device


def setup_plot(steps_number):

    """Create the real-time diagnostic figure and its Matplotlib artists."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
    
    # Создание линий
    line1, = ax1.plot(torch.arange(1, steps_number + 1), torch.zeros(steps_number), 'r.', label='Times')
    line2, = ax1.plot(torch.arange(1, steps_number + 1), torch.zeros(steps_number), label="Time's Fit")
    line3, = ax2.plot(torch.arange(1, steps_number + 1), torch.zeros(steps_number), label='Bond')
    line4, = ax3.plot(torch.arange(1, steps_number + 1), torch.zeros(steps_number), label="State's norm")
    
    # Создание текстовых элементов
    text1 = ax1.text(0.5, 0.85, '', transform=ax1.transAxes, ha='center')
    text2 = ax2.text(0.2, 0.9, '', transform=ax2.transAxes, ha='center')
    text3 = ax3.text(0.2, 0.8, '', transform=ax3.transAxes, ha='center')
    
    # Настройка сетки для всех осей
    for ax in [ax1, ax2, ax3]:
        ax.grid(which='major', alpha=0.6)
        ax.grid(which='minor', alpha=0.2)
        ax.minorticks_on()
        ax.legend()
    
    return fig, (ax1, ax2, ax3), (line1, line2, line3, line4), (text1, text2, text3)


def update_plot(i, times_container, bonds, norms, steps_number, lines, axes, texts, fig, bond_limit, discrepancy, cut_time_base=10):

    # Вычисление cut_time
    #cut_time = cut_time_base + i // 10
    
    # Полиномиальная аппроксимация
    """Update elapsed-time fit, bond and norm curves in the diagnostic figure."""
    if i < 8:
        poly_coeffs = np.polyfit(torch.arange(1, i + 1), times_container, 2)
        #print(len(torch.arange(1, i + 1)), len(times_container))
    else:
        poly_coeffs = np.polyfit(torch.arange(i - 8, i), 
                                times_container[- 8:], 2)
        #print(len(torch.arange(i - (cut_time - 1), i+1)), len(times_container[i - cut_time:i]))
    
    poly_func = np.poly1d(poly_coeffs)
    
    # Обновление данных на графиках
    lines[0].set_ydata(times_container + [0] * (steps_number - len(times_container)))
    lines[1].set_ydata(poly_func(torch.arange(1, steps_number + 1)))
    lines[2].set_ydata(bonds + [0] * (steps_number - len(times_container)))
    lines[3].set_ydata(norms + [0] * (steps_number - len(times_container)))
    
    # Автомасштабирование осей
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    
    # Вычисление времени последней итерации
    if len(times_container) > 1:
        last_time = np.round(times_container[-1] - times_container[-2])
    else:
        last_time = 0
    
    # Обновление текстовой информации
    total_time_estimate = np.round(poly_func(steps_number))
    remaining_time = np.round(total_time_estimate - times_container[-1]) if i > 0 else 0
    
    texts[0].set_text(
        f'Общее время выполнения {total_time_estimate} секунды\n'
        f'Оставшееся время {remaining_time} секунды\n'
        f'Кол.вып.ит. {i}\n'
        f'Время посл. ит. {last_time}'
    )
    texts[1].set_text(
        f'Бонд {bonds[-1]} \n'
        f'Ограничение\nбонда {bond_limit}'              
    )
    texts[2].set_text(
        f'Норма\nсостояния\n{norms[-1]}\n'
        f'Невязка {discrepancy}\n'
    )
    
    # Установка размера шрифта
    for text in texts:
        text.set_fontsize(8)
    
    # Обновление отображения
    clear_output(wait=True)
    display(fig)
    plt.pause(0.0001)  # Пауза для обновления графика
    
    return poly_func, total_time_estimate, remaining_time, last_time

    
def compute_discrepancy(operator, solution, right_part):
    """Compute the relative global residual ||A x - b|| / ||b|| in tensor-train form."""
    return(compute_norm(mps_sum(mpo_x_mps(operator, solution), mps_mul_num(right_part, -1)))/compute_norm(right_part))

    
def update_lists(list_for_matrices, times_container, norms, bonds, solution, start_time, initial_guess, bond_limit, discrepancy):
    """Append or roll back trajectory data after an adaptive-rank decision."""
    if initial_guess[0] == "adaptive" and discrepancy > initial_guess[4] and bond_limit < initial_guess[6]:
        list_for_matrices_changed = list_for_matrices.copy()+[None]
        times_container_changed = times_container.copy()+[None]
        norms_changed = norms.copy()+[None]
        bonds_changed = bonds.copy()+[None]

        return list_for_matrices_changed[:-min(initial_guess[5]+1, len(list_for_matrices))], times_container_changed[:-min(initial_guess[5]+1, len(times_container)+1)], norms_changed[:-min(initial_guess[5]+1, len(norms)+1)], bonds_changed[:-min(initial_guess[5]+1, len(bonds)+1)]
    else:
        return list_for_matrices+[solution], times_container+[time.time() - start_time], norms+[np.round(compute_norm(solution), 2)], bonds+[get_bond(solution)]

    
def update_bond_limit(initial_guess, bond_limit, discrepancy):
    """Increase the permitted bond dimension when adaptive residual criteria are triggered."""
    if initial_guess[0] == "adaptive" and discrepancy > initial_guess[4] and bond_limit < initial_guess[6]:
        return bond_limit + initial_guess[3]
    else:
        return bond_limit
    

def update_iteration(i, initial_guess, bond_limit, discrepancy):
    """Rewind the current time-step index for the adaptive-rank rollback strategy."""
    if initial_guess[0] == "adaptive" and discrepancy > initial_guess[4] and bond_limit < initial_guess[6]:
        return i - min(initial_guess[5]+1, i)
    else:
        return i
    
    
def setup_guess(initial_guess, state):
    """Choose the initial MPS approximation according to the configured guess strategy.
    
    Parameters:
        initial_guess: Initial-state strategy; may be a named mode or a configuration list.
        state: MPS representing the initial or current state.
    
    Returns:
        MPS to use as the starting point for the next solve."""
    if initial_guess == "best":
        guess = state
        return guess
    elif isinstance(initial_guess, list):
        if initial_guess[0] == "approx":
            guess = mps_compression(state, initial_guess[1], initial_guess[2])
            return guess
        if initial_guess[0] == "adaptive":
            # initial_guess[1] - эпсилон сжатия, initial_guess[2] - бонд сжатия, initial_guess[3] - шаг увеличения бонда, 
            # initial_guess[4] - порог, при котором увеличивается бонд, initial_guess[5] - откат по шагам назад
            guess = mps_compression(state, initial_guess[1], initial_guess[2]) # initial_guess[3], initial_guess[4], initial_guess[5])
            return guess
        else:
            raise ValueError("Invalid initial_guess list format")
    else:
        raise ValueError("Invalid initial_guess type")
    
    
def evolute_euler(state, operator, steps_number, solver, initial_guess="best", eps=1e-6, chart=True, bond_limit=1000, sweep_number=1, renormalize=True, way=False):
    # Keep the original public API while moving Euler evolution to its module.
    """Compatibility time-stepping wrapper for the Euler solver.
    
    Parameters:
        state: MPS representing the initial or current state.
        operator: MPO operator supplied to the method.
        steps_number: Number of time steps to advance.
        solver: Callable that solves one time step.
        initial_guess: Initial-state strategy; may be a named mode or a configuration list.
        eps: Truncation tolerance used by the implementation (see singular-value rule).
        chart: Whether to draw diagnostic plots during/after evolution.
        bond_limit: Upper bound on the virtual bond dimension after splitting or compression.
        sweep_number: Number of complete left-right/right-left optimisation sweeps.
        renormalize: Whether to rescale the resulting state to the input norm.
        way: Optional output path prefix for periodically pickled state lists.
    
    Returns:
        MPS trajectory, optionally together with a diagnostics dictionary."""
    from ImplicitEuler_torch import evolute_euler as _evolute_euler
    return _evolute_euler(state, operator, steps_number, solver=solver,
                         initial_guess=initial_guess, eps=eps, chart=chart,
                         bond_limit=bond_limit, sweep_number=sweep_number,
                         renormalize=renormalize, way=way)


def mps_scalar_multiplication(mps_1, mps_2):

    """Contract two MPS networks to obtain the complex inner product <mps_1|mps_2>."""
    if not isinstance(mps_1, list) or not isinstance(mps_2, list):
        raise TypeError("Аргументы должны быть списками")
    if len(mps_1) != len(mps_2):
        raise ValueError("Длины двух mps должны быть одинаковы")

    mps1 = mps_1
    mps2 = mps_2
    sh1 = mps1[0].shape
    sh2 = mps2[0].shape
    nnn = torch.einsum('ijk,mjl->imkl', torch.conj(mps1[0]), mps2[0]).reshape(sh1[0] * sh2[0], sh1[2] * sh2[2])
    for i in range(1, len(mps1)):
        sh1 = mps1[i].shape
        sh2 = mps2[i].shape
        nnn = nnn @ torch.einsum('ijk,mjl->imkl', torch.conj(mps1[i]), mps2[i]).reshape(sh1[0]*sh2[0], sh1[2]*sh2[2])
    nnn = nnn.reshape(-1)[0]
    return nnn


def mpo_transpose(mpo):
    """Exchange the input and output physical indices in every MPO core."""
    transposed_mpo = []
    for tensor in mpo:
        transposed_mpo.append(torch.permute(tensor, (0,2,1,3)))
    return transposed_mpo


def swap_first_and_last_axes_in_mpo(mpo):
    """Reverse the left and right virtual-index positions of each MPO tensor."""
    new_mpo = []
    for tensor in mpo:
        new_mpo.append(torch.permute(tensor, (3, 1, 2, 0)))
    return new_mpo


def swap_first_and_last_axes_in_mps(mps):
    """Reverse the left and right virtual-index positions of each MPS tensor."""
    new_mps = []
    for tensor in mps:
        new_mps.append(torch.permute(tensor, (2, 1, 0)))
    return new_mps


def coarsen_two_mode_mps(mps, delta_n):
    """Restrict the outermost binary physical legs of a two-mode MPS to zero."""
    new_mps = mps.copy()
    for i in range(delta_n):
        new_mps[i] = new_mps[i][:,0:1,:]
        new_mps[-i-1] = new_mps[-i-1][:,0:1,:]
    return new_mps


def coarsen_two_mode_mpo(mpo, delta_n):
    """Restrict the outermost binary input/output legs of a two-mode MPO to zero."""
    new_mpo = mpo.copy()
    for i in range(delta_n):
        new_mpo[i] = new_mpo[i][:,0:1,0:1,:]
        new_mpo[-i-1] = new_mpo[-i-1][:,0:1,0:1,:]
    return new_mpo


def coarsen_the_mps(mps, delta_n):
    """Restrict the final delta_n binary physical legs of an MPS to zero."""
    new_mps = mps.copy()
    for i in range(delta_n):
        new_mps[-i-1] = new_mps[-i-1][:,0:1,:]
    return new_mps


def coarsen_the_mpo(mpo, delta_n):
    """Restrict the final delta_n binary physical legs of an MPO to zero."""
    new_mpo = mpo.copy()
    for i in range(delta_n):
        new_mpo[-i-1] = new_mpo[-i-1][:,0:1,0:1,:]
    return new_mpo


def linear_interpolate_mps(mps, n):

    """Iteratively refine an MPS grid using a shift MPO and additional binary sites."""
    def shift_operator_mpo(n):
        """Build the binary shift MPO used by the MPS interpolation routine."""
        if n < 2:
            raise ValueError("n must be >= 2")

        first = torch.tensor([1,0,0,1,0,0,1,0]).reshape(1,2,2,2)
        bulk  = torch.tensor([1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0]).reshape(2,2,2,2)
        last  = torch.tensor([0,1,0,0,0,0,1,0]).reshape(2,2,2,1)

        structure_mpo = [first] + (n-2)*[bulk] + [last]
        return structure_mpo

    def unite_two_mps(mps1, mps2):
        """Append a binary marker site to combine two MPS branches."""
        return mps_sum(mps1+[torch.tensor((1,0)).reshape(1,2,1)], mps2+[torch.tensor((0,1)).reshape(1,2,1)])

    av = mps
    for i in range(n):
        bv = mps_mul_num(mps_sum(mpo_x_mps(shift_operator_mpo(len(av)), av), av), 0.5)
        av = unite_two_mps(av, bv)

    return av


def find_reduced_density_matrix2(mps, position, length, n_compression):
    """Form a reduced density matrix for a two-mode MPS, optionally coarse-graining sites."""
    l = len(mps)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]
    state_compressed_part = state_part[:len(state_part)-n_compression]
    compressed_part = state_part[len(state_part)-n_compression:]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[-1], torch.conj(env_part[-1]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], torch.conj(env_part[-i-1]))
    else:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[0], torch.conj(env_part[0]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], torch.conj(env_part[i]))

    # Усреднение (угрубление) матрицы плотности состояния 
    for i in range(len(compressed_part)):
        compressed_part[i] = torch.einsum("ijk -> ik", compressed_part[i])
        compressed_part[i] = torch.einsum("ab, cd -> acbd", compressed_part[i], torch.conj(compressed_part[i]))

    if position == 0:
        first_part = first_env_step
        for i in range(len(compressed_part)):
            first_part = torch.einsum("abcd, cdjk -> abjk", compressed_part[-1-i], first_part)
        
        first_part = torch.einsum("abc, ijk, ckgh -> aibjgh", state_compressed_part[-1], torch.conj(state_compressed_part[-1]), first_part)
        for i in range(1, len(state_compressed_part)):
            first_part = torch.einsum("abc, ijk, ckghml -> aibgjhml", state_compressed_part[-i-1], torch.conj(state_compressed_part[-i-1]), first_part)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])
    
    else:
        first_part = torch.einsum("abcd, cjk, dgh -> abjgkh", first_env_step, state_compressed_part[0], torch.conj(state_compressed_part[0]))
        for i in range(1, len(state_compressed_part)):
            first_part = torch.einsum("abcdef, ejk, fgh -> abcjdgkh", first_part, state_compressed_part[i], torch.conj(state_compressed_part[i]))
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        for i in range(len(compressed_part)):
            first_part = torch.einsum("abcdef, efjk -> abcdjk", first_part, compressed_part[i])
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])


def find_reduced_density_matrix3(mps, position, length):
    """Construct a reduced density matrix for the selected mode of a three-mode MPS.
    
    Parameters:
        mps: MPS tensor list with shape (left_bond, physical, right_bond).
        position: Index of the selected physical mode.
        length: Number of binary sites encoding a mode.
    
    Returns:
        Dense reduced density matrix for the selected mode.
    
    Notes:
        The module contains two definitions with this name; Python uses the last definition."""
    l = len(mps)
    state_part = mps[position*length:(position+1)*length]

    # Свертка состояния окружения
    if position == 0:
        env_part = mps[length:]
        first_env_step = torch.einsum('abc, ibk -> aick', env_part[-1], torch.conj(env_part[-1]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum('abcd, ika, jkb -> ijcd', first_env_step, env_part[-i-1], torch.conj(env_part[-i-1]))

    elif position == 2:
        env_part = mps[:2*length]
        first_env_step = torch.einsum('abc, ibk -> aick', env_part[0], torch.conj(env_part[0]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum('abcd, cij, dik -> abjk', first_env_step, env_part[i], torch.conj(env_part[i]))

    else:
        env_part_left = mps[:length]
        env_part_right = mps[2*length:]

        first_env_step = torch.einsum('abc, ibk -> aick', env_part_right[-1], torch.conj(env_part_right[-1]))
        for i in range(1, len(env_part_right)):
            first_env_step = torch.einsum('abcd, ika, jkb -> ijcd', first_env_step, env_part_right[-i-1], torch.conj(env_part_right[-i-1]))

        second_env_step = torch.einsum('abc, ibk -> aick', env_part_left[0], torch.conj(env_part_left[0]))
        for i in range(1, len(env_part_left)):
            second_env_step = torch.einsum('abcd, cij, dik -> abjk', second_env_step, env_part_left[i], torch.conj(env_part_left[i]))

    if position in (0, 1):
        first_part = torch.einsum('abc, ijk, ckgh -> aibjgh', state_part[-1], torch.conj(state_part[-1]), first_env_step)
        for i in range(1, len(state_part)):
            first_part = torch.einsum('abc, ijk, ckghml -> aibgjhml', state_part[-i-1], torch.conj(state_part[-i-1]), first_part)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        if position == 1:
            first_part = torch.einsum('abcd, cdefgh -> abefgh', second_env_step, first_part)

        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])

    else:
        first_part = torch.einsum('abcd, cjk, dgh -> abjgkh', first_env_step, state_part[0], torch.conj(state_part[0]))
        for i in range(1, len(state_part)):
            first_part = torch.einsum('abcdef, ejk, fgh -> abcjdgkh', first_part, state_part[i], torch.conj(state_part[i]))
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])



def diag_mpo_from_mps(mps):
    """Embed the amplitudes of an MPS into the diagonal of an MPO."""
    mpo = []
    for A in mps:
        Dl, d, Dr = A.shape
        W = torch.zeros((Dl, d, d, Dr), dtype=A.dtype, device=A.device)
        idx = torch.arange(d, device=A.device)
        W[:, idx, idx, :] = A
        mpo.append(W)
    return mpo


def annihilate_operator_mpo(n, dtype=torch.float64):
    """Build the Fock-basis bosonic annihilation operator in binary MPO form."""
    if n < 2:
        raise ValueError("n must be >= 2")

    first = torch.tensor([1,0,0,1,0,0,1,0], dtype=dtype).reshape(1,2,2,2)
    bulk  = torch.tensor([1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0], dtype=dtype).reshape(2,2,2,2)
    last  = torch.tensor([0,1,0,0,0,0,1,0], dtype=dtype).reshape(2,2,2,1)

    structure_mpo = [first] + (n-2)*[bulk] + [last]

    vec = torch.sqrt(torch.arange(2**n, dtype=torch.float64)).type(dtype)
    vec_mps = vec_to_mps(vec, eps=1e-11)
    diag_mpo = diag_mpo_from_mps(vec_mps)

    return mpo_multiplication(structure_mpo, diag_mpo)


def double_annihilate_operator_mpo(n, dtype=torch.float64):
    """Build the operator a**2 in a binary-encoded Fock basis."""
    if n < 2:
        raise ValueError("n must be >= 2")

    A = torch.tensor([[1,0],[0,1]], dtype=dtype)
    B = torch.tensor([[0,1],[0,0]], dtype=dtype)
    C = torch.zeros((2,2), dtype=dtype)
    D = torch.tensor([[0,0],[1,0]], dtype=dtype)

    first = torch.stack([A, B], dim=-1).reshape(1,2,2,2)
    block1 = torch.stack([A, B], dim=-1)
    block2 = torch.stack([C, D], dim=-1)
    bulk = torch.stack([block1, block2], dim=0)
    last = torch.stack([C, A], dim=0).reshape(2,2,2,1)

    structure_mpo = [first] + (n-2)*[bulk] + [last]

    k = torch.arange(2**n, dtype=torch.float64)
    vec = torch.sqrt(k * torch.clamp(k - 1, min=0)).to(dtype)
    vec_mps = vec_to_mps(vec, eps=1e-11)
    diag_mpo = diag_mpo_from_mps(vec_mps)

    return mpo_multiplication(structure_mpo, diag_mpo)


def create_operator_mpo(n, dtype=torch.float64):
    """Build a bosonic creation MPO by Hermitian transposition of annihilation cores."""
    annihilate_mpo = annihilate_operator_mpo(n, dtype=dtype)
    return [torch.conj(A).swapaxes(1, 2) for A in annihilate_mpo]


def double_create_operator_mpo(n, dtype=torch.float64):
    """Build the (a_dagger)**2 MPO from its annihilation counterpart."""
    double_annihilate_mpo = double_annihilate_operator_mpo(n, dtype=dtype)
    return [torch.conj(A).swapaxes(1, 2) for A in double_annihilate_mpo]


def calculate_photon_number_in_Fock_basis(mps, position, length):
    """Compute the mean occupation of one selected mode in a two-mode Fock MPS."""
    l = len(mps)
    photon_number_op = [torch.as_tensor(t, dtype=mps[0].dtype, device=mps[0].device) for t in x_mpo(length)]
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[-1], torch.conj(env_part[-1]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], torch.conj(env_part[-i-1]))
    else:
        first_env_step = torch.einsum("abc, ibk -> aick", env_part[0], torch.conj(env_part[0]))
        for i in range(1, len(env_part)):
            first_env_step = torch.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], torch.conj(env_part[i]))

    # Свертка матрицы плотности состояния с оператором числа фотонов
    if position == 0:
        first_state_step = torch.einsum("abc, ibjk, ejf -> aieckf", state_part[0], photon_number_op[0], torch.conj(state_part[0]))
        for i in range(1, len(state_part)):
            first_state_step = torch.einsum("abcdef, djk, ejop, fow -> abckpw", first_state_step, state_part[i], photon_number_op[i], torch.conj(state_part[i]))
    else:
        first_state_step = torch.einsum("abc, ibjk, ejf -> aieckf", state_part[-1], photon_number_op[-1], torch.conj(state_part[-1]))
        for i in range(1, len(state_part)):
            first_state_step = torch.einsum("abcdef, gja, hjob, woc -> ghwdef", first_state_step, state_part[-i-1], photon_number_op[-i-1], torch.conj(state_part[-i-1]))

    # Вычисление следа матрицы плотности
    if position == 0:
        rho_trace_first_step = torch.einsum("abc, ibk -> aick", state_part[0], torch.conj(state_part[0]))
        for i in range(1, len(state_part)):
            rho_trace_first_step = torch.einsum("abcd, cij, dif -> abjf", rho_trace_first_step, state_part[i], torch.conj(state_part[i]))
    else:
        rho_trace_first_step = torch.einsum("abc, ibk -> aick", state_part[-1], torch.conj(state_part[-1]))
        for i in range(1, len(state_part)):
            rho_trace_first_step = torch.einsum("abcd, ika, jkb -> ijcd", rho_trace_first_step, state_part[-i-1], torch.conj(state_part[-i-1]))

    if position == 1:
        common_trace = torch.einsum("abcd, cdij -> abij", first_env_step, rho_trace_first_step).reshape(-1)
        return torch.einsum("abcd, cfdklm -> afbklm", first_env_step, first_state_step).reshape(-1)/common_trace
    else:
        common_trace = torch.einsum("abcd, cdij -> abij", rho_trace_first_step, first_env_step).reshape(-1)
        return torch.einsum("abcd, ijkahb -> ijkchd", first_env_step, first_state_step).reshape(-1)/common_trace


def calculate_photon_number_in_Fock_basis_3modes(mps, position, length):
    """Compute the mean occupation of one mode in a three-mode Fock MPS.
    
    Notes:
        The module contains two definitions with this name; Python uses the last definition."""
    l = len(mps)
    photon_number_op = [torch.as_tensor(t, dtype=mps[0].dtype, device=mps[0].device) for t in x_mpo(length)]
    state_part = mps[position*length:(position+1)*length]
    left_env_part = mps[:position*length]
    right_env_part = mps[(position+1)*length:]

    # Свертка левого окружения
    if len(left_env_part) > 0:
        first_left_env_step = torch.einsum("abc, ibk -> aick", left_env_part[0], torch.conj(left_env_part[0]))
        for i in range(1, len(left_env_part)):
            first_left_env_step = torch.einsum("abcd, cij, dik -> abjk", first_left_env_step, left_env_part[i], torch.conj(left_env_part[i]))
    else:
        first_left_env_step = torch.ones((1, 1, 1, 1), dtype=mps[0].dtype, device=mps[0].device)

    # Свертка правого окружения
    if len(right_env_part) > 0:
        first_right_env_step = torch.einsum("abc, ibk -> aick", right_env_part[-1], torch.conj(right_env_part[-1]))
        for i in range(1, len(right_env_part)):
            first_right_env_step = torch.einsum("abcd, ika, jkb -> ijcd", first_right_env_step, right_env_part[-i-1], torch.conj(right_env_part[-i-1]))
    else:
        first_right_env_step = torch.ones((1, 1, 1, 1), dtype=mps[0].dtype, device=mps[0].device)

    # Свертка матрицы плотности состояния с оператором числа фотонов
    first_state_step = torch.einsum("abc, ibjk, ejf -> aieckf", state_part[0], photon_number_op[0], torch.conj(state_part[0]))
    for i in range(1, len(state_part)):
        first_state_step = torch.einsum("abcdef, djk, ejop, fow -> abckpw", first_state_step, state_part[i], photon_number_op[i], torch.conj(state_part[i]))

    # Вычисление следа матрицы плотности
    rho_trace_first_step = torch.einsum("abc, ibk -> aick", state_part[0], torch.conj(state_part[0]))
    for i in range(1, len(state_part)):
        rho_trace_first_step = torch.einsum("abcd, cij, dif -> abjf", rho_trace_first_step, state_part[i], torch.conj(state_part[i]))

    # Объединение левого и правого окружений
    common_trace = torch.einsum("abcd, cdef, efgh -> abgh", first_left_env_step, rho_trace_first_step, first_right_env_step).reshape(-1)

    return torch.einsum("abcd, cidjke, jefg -> abikfg", first_left_env_step, first_state_step, first_right_env_step).reshape(-1)/common_trace


def mps_mpo_mps(psi_one, mpo, psi_two=None):
    """Contract <psi_one|MPO|psi_two> without explicitly creating a dense vector."""
    if psi_two is None:
        psi_two = psi_one

    dtype = torch.promote_types(torch.promote_types(psi_one[0].dtype, psi_two[0].dtype), mpo[0].dtype)
    env = torch.ones((1, 1, 1), dtype=dtype, device=psi_one[0].device)

    for A, W, B in zip(psi_one, mpo, psi_two):
        A = A.to(dtype)
        B = B.to(dtype)
        W = W.to(dtype)

        ra, rw, rb = env.shape
        ra_next, rw_next, rb_next = A.shape[2], W.shape[3], B.shape[2]

        temp = (env.permute(1, 0, 2).reshape(rw*ra, rb) @ B.reshape(rb, B.shape[1]*rb_next)).reshape(rw, ra, B.shape[1], rb_next)

        temp = (W.permute(1, 3, 0, 2).reshape(W.shape[1]*rw_next, rw*W.shape[2]) @ temp.permute(0, 2, 1, 3).reshape(rw*W.shape[2], ra*rb_next)).reshape(W.shape[1], rw_next, ra, rb_next).permute(2, 0, 1, 3)

        env = (A.conj().permute(2, 0, 1).reshape(ra_next, ra*A.shape[1]) @ temp.reshape(ra*A.shape[1], rw_next*rb_next)).reshape(ra_next, rw_next, rb_next)

    return env[0, 0, 0]


def Hmps_Hmps(psi_one, mpo, psi_two=None):
    """Contract the MPS/MPO network used for Hamiltonian-squared overlap expressions."""
    if psi_two is None:
        psi_two = psi_one

    dtype = torch.promote_types(torch.promote_types(psi_one[0].dtype, psi_two[0].dtype), mpo[0].dtype)
    env = torch.ones((1, 1, 1, 1), dtype=dtype, device=psi_one[0].device)

    for A, W, B in zip(psi_one, mpo, psi_two):
        A = A.to(dtype)
        B = B.to(dtype)
        W = W.to(dtype)

        ra, rw_bra, rw_ket, rb = env.shape
        ra_next, rw_next, rb_next = A.shape[2], W.shape[3], B.shape[2]

        temp = (env.reshape(ra*rw_bra*rw_ket, rb) @ B.reshape(rb, B.shape[1]*rb_next)).reshape(ra, rw_bra, rw_ket, B.shape[1], rb_next)

        temp = (temp.permute(0, 1, 4, 2, 3).reshape(ra*rw_bra*rb_next, rw_ket*W.shape[2]) @ W.permute(0, 2, 1, 3).reshape(rw_ket*W.shape[2], W.shape[1]*rw_next)).reshape(ra, rw_bra, rb_next, W.shape[1], rw_next).permute(0, 1, 3, 4, 2)

        temp = (temp.permute(0, 3, 4, 1, 2).reshape(ra*rw_next*rb_next, rw_bra*W.shape[1]) @ W.conj().reshape(rw_bra*W.shape[1], W.shape[2]*rw_next)).reshape(ra, rw_next, rb_next, W.shape[2], rw_next).permute(0, 3, 4, 1, 2)

        env = (A.conj().permute(2, 0, 1).reshape(ra_next, ra*A.shape[1]) @ temp.reshape(ra*A.shape[1], rw_next*rw_next*rb_next)).reshape(ra_next, rw_next, rw_next, rb_next)

    return env[0, 0, 0, 0]


def compute_CN_residual(psi_old, psi_new, hamiltonian, delta_t):
    """Evaluate a Crank–Nicolson defect estimate from MPS/MPO scalar contractions.
    
    Parameters:
        psi_old: Value used by this routine; see the function signature and body.
        psi_new: Value used by this routine; see the function signature and body.
        hamiltonian: Value used by this routine; see the function signature and body.
        delta_t: Time-step size in the same units as the Hamiltonian.
    
    Notes:
        Formula uses the supplied MPO as Hamiltonian and is distinct from the global discrete linear-system residual."""
    norm1 = compute_norm(mps_sum(psi_new, mps_mul_num(psi_old, -1)))**2
    norm2 = Hmps_Hmps(psi_new, hamiltonian, psi_new)
    norm3 = Hmps_Hmps(psi_old, hamiltonian, psi_old)
    norm4 = 2*torch.real(Hmps_Hmps(psi_old, hamiltonian, psi_new))
    norm5 = 2*torch.imag(mps_mpo_mps(psi_old, hamiltonian, psi_new))
    norm_psi_old = compute_norm(psi_old)

    return torch.sqrt(norm1 + 0.25*(norm2 + norm3 + norm4)*delta_t**2 + delta_t*norm5).real/norm_psi_old


def calculate_photon_number_in_Fock_basis_3modes(mps, position, lengths, little_endian=False):
    """Compute the mean occupation of one mode in a three-mode Fock MPS.
    
    Notes:
        The module contains two definitions with this name; Python uses the last definition."""
    if isinstance(lengths, int):
        lengths = (lengths,)*3

    if len(lengths) != 3 or sum(lengths) != len(mps):
        raise ValueError('Некорректные длины мод')
    if position not in (0, 1, 2):
        raise ValueError('position должен быть 0, 1 или 2')
    if any(length <= 0 for length in lengths):
        raise ValueError('Длины мод должны быть положительными')

    start = sum(lengths[:position])
    end = start + lengths[position]

    environment = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)

    for i, tensor in enumerate(mps):
        a0, a1 = tensor[:, 0, :], tensor[:, 1, :]

        env0 = a0.T @ environment @ a0.conj()
        env1 = a1.T @ environment @ a1.conj()

        if start <= i < end:
            k = i-start if little_endian else end-1-i
            term = (2**k)*env1

            if i == start:
                photon_environment = term
            else:
                photon_environment = (
                    a0.T @ photon_environment @ a0.conj()
                    + a1.T @ photon_environment @ a1.conj()
                    + term
                )

        elif i >= end:
            photon_environment = (
                a0.T @ photon_environment @ a0.conj()
                + a1.T @ photon_environment @ a1.conj()
            )

        environment = env0 + env1

    return (photon_environment.reshape(-1) / environment.reshape(-1)).real


def find_reduced_density_matrix3(mps, position, lengths):
    """Construct a reduced density matrix for the selected mode of a three-mode MPS.
    
    Parameters:
        mps: MPS tensor list with shape (left_bond, physical, right_bond).
        position: Index of the selected physical mode.
        lengths: Value used by this routine; see the function signature and body.
    
    Returns:
        Dense reduced density matrix for the selected mode.
    
    Notes:
        The module contains two definitions with this name; Python uses the last definition."""
    if isinstance(lengths, int):
        lengths = (lengths, lengths, lengths)

    if len(lengths) != 3 or any(n <= 0 for n in lengths) or sum(lengths) != len(mps):
        raise ValueError('Некорректные длины мод')
    if position not in (0, 1, 2):
        raise ValueError('position должен быть равен 0, 1 или 2')

    start = sum(lengths[:position])
    end = start + lengths[position]
    state_part = mps[start:end]
    left_env_part = mps[:start]
    right_env_part = mps[end:]

    # Свертка левого окружения
    first_left_env_step = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    for tensor in left_env_part:
        a0, a1 = tensor[:, 0, :], tensor[:, 1, :]
        first_left_env_step = (a0.T @ first_left_env_step @ torch.conj(a0)
                               + a1.T @ first_left_env_step @ torch.conj(a1))

    # Свертка правого окружения
    first_right_env_step = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    for tensor in right_env_part[::-1]:
        a0, a1 = tensor[:, 0, :], tensor[:, 1, :]
        first_right_env_step = (a0 @ first_right_env_step @ torch.conj(a0).T
                                + a1 @ first_right_env_step @ torch.conj(a1).T)

    # Объединение физических индексов выбранной моды
    first_part = state_part[0]
    for tensor in state_part[1:]:
        sh = first_part.shape
        first_part = (first_part.reshape(-1, sh[-1]) @ tensor.reshape(tensor.shape[0], -1)).reshape(
            sh[0], sh[1]*tensor.shape[1], tensor.shape[-1]
        )

    # Свертка состояния с окружениями
    sh = first_part.shape
    weighted_part = (first_left_env_step.T @ first_part.reshape(sh[0], -1)).reshape(sh)
    weighted_part = (weighted_part.reshape(-1, sh[-1]) @ first_right_env_step).reshape(sh)

    first_part = first_part.permute(1, 0, 2).reshape(sh[1], -1)
    weighted_part = weighted_part.permute(1, 0, 2).reshape(sh[1], -1)

    return weighted_part @ torch.conj(first_part).T


def create_two_basis_vectors():
    """Construct two Hamiltonian-powered trial vectors from variables in notebook scope.
    
    Notes:
        This helper relies on global names hamiltonian, psi and eps_basis; it is not independently callable without defining them."""
    Hpsi = mpo_x_mps_zip_up(
        hamiltonian, psi, eps=eps_basis, bond_limit=150
    )

    H2psi = mpo_x_mps_zip_up(
        hamiltonian, Hpsi, eps=eps_basis, bond_limit=150
    )

    return [Hpsi, H2psi]


def create_basis_vector(psi, hamiltonian, eps_basis=1e-6, bond_limit=150):
    """Construct a compressed trial vector H|psi> for Krylov-like augmentation."""
    Hpsi = mpo_x_mps_zip_up(
        hamiltonian, psi, eps=eps_basis, bond_limit=bond_limit
    )
    return [Hpsi]


def mpo_x_mps_zip_up(mpo, mps, eps=1e-6, bond_limit=500):
    """Apply an MPO to an MPS with on-the-fly SVD truncation (zip-up contraction)."""
    compressed_mps = []
    dtype = torch.promote_types(mpo[0].dtype, mps[0].dtype)
    tensor = torch.ones((1, 1, 1), dtype=dtype, device=mps[0].device)

    for i in range(len(mps)):
        w, a = mpo[i].to(dtype), mps[i].to(dtype)

        tmp = torch.einsum('pla, atb -> pltb', tensor, a)
        tmp = torch.einsum('pltb, lstr -> psrb', tmp, w)
        sh = tmp.shape

        if i < len(mps)-1:
            u, s, vh = torch.linalg.svd(
                tmp.reshape(sh[0]*sh[1], sh[2]*sh[3]),
                full_matrices=False
            )

            s2 = s**2
            relative_tail2 = torch.cumsum(s2.flip(0), 0).flip(0) / s2.sum().clamp_min(torch.finfo(s.dtype).tiny)
            r = int((relative_tail2 > eps**2).sum().item())
            bond = min(max(r, 1), bond_limit)

            compressed_mps.append(u[:, :bond].reshape(sh[0], sh[1], bond))
            tensor = (s[:bond, None]*vh[:bond, :]).reshape(bond, sh[2], sh[3])

        else:
            compressed_mps.append(tmp.reshape(sh[0], sh[1], 1))

    return compressed_mps



def calculate_photon_number_statistics(set_of_solutions, n_modes, x0_modes, step_modes, sample_step=10):
    """
    Calculate mean photon numbers for each mode using MPS-MPO-MPS contractions.

    Parameters
    ----------
    set_of_solutions : list
        List of MPS states at different time steps.
    n_modes : list
        Number of binary MPS sites for each mode.
    x0_modes : list
        Left boundaries of the quadrature grids.
    step_modes : list
        Quadrature grid spacings.
    sample_step : int
        Interval between sampled time steps.

    Returns
    -------
    photon_number_stats : list
        Mean photon numbers for each mode at the sampled time steps.
    """

    if not (len(n_modes) == len(x0_modes) == len(step_modes)):
        raise ValueError("Mode parameter lists must have equal lengths.")

    dtype = set_of_solutions[0][0].dtype
    device = set_of_solutions[0][0].device

    # Construct photon-number MPOs for all modes
    number_ops = []

    for mode in range(len(n_modes)):
        number_mpo = []

        for j in range(len(n_modes)):
            if j == mode:
                number_mpo += photon_number_operator_mpo(
                    n_modes[j], x0_modes[j], step_modes[j], dtype=dtype
                )
            else:
                number_mpo += one_mpo(n_modes[j], dtype=dtype)

        number_ops.append([tensor.to(device) for tensor in number_mpo])

    # Calculate photon-number expectation values
    photon_number_stats = [[] for _ in range(len(n_modes))]

    for i in tqdm(range(0, len(set_of_solutions), sample_step)):
        psi = set_of_solutions[i]
        norm2 = mps_scalar_multiplication(psi, psi).real

        for mode in range(len(n_modes)):
            photon_number = (
                mps_mpo_mps(psi, number_ops[mode]).real / norm2 - 0.5
            )

            photon_number_stats[mode].append(photon_number.item())

    return photon_number_stats
