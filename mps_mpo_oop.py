"""Небольшие классы MPS и MPO поверх списков тензоров PyTorch.

MPS core shape: (left_bond, physical, right_bond).
MPO core shape: (left_bond, physical_out, physical_in, right_bond).
"""
from numbers import Number
import torch


def _is_number(x):
    """Return whether x is a Python scalar or a zero-dimensional torch tensor."""
    return isinstance(x, Number) or (isinstance(x, torch.Tensor) and x.ndim == 0)


def _tt_sum(first, second):
    """Form the exact tensor-train sum of two lists of compatible MPS or MPO cores."""
    if len(first) != len(second):
        raise ValueError("Число тензоров должно совпадать")
    if not first:
        raise ValueError("Пустые MPS/MPO не поддерживаются")
    if len(first) == 1:
        return [first[0] + second[0]]

    result = [torch.cat([first[0], second[0]], dim=-1)]
    for a, b in zip(first[1:-1], second[1:-1]):
        if a.shape[1:-1] != b.shape[1:-1]:
            raise ValueError("Физические размерности должны совпадать")
        t = torch.zeros(
            (a.shape[0] + b.shape[0], *a.shape[1:-1], a.shape[-1] + b.shape[-1]),
            dtype=torch.promote_types(a.dtype, b.dtype), device=a.device,
        )
        t[:a.shape[0], ..., :a.shape[-1]] = a
        t[a.shape[0]:, ..., a.shape[-1]:] = b
        result.append(t)
    result.append(torch.cat([first[-1], second[-1]], dim=0))
    return result


def mps_sum(psi_one, psi_two):
    """Return a list of cores representing the exact sum of two MPS states."""
    return _tt_sum(psi_one, psi_two)


def mpo_sum(mpo_one, mpo_two):
    """Return a list of cores representing the exact sum of two MPO operators."""
    return _tt_sum(mpo_one, mpo_two)


def mps_mul_num(mps, num):
    """Scale an MPS via its first core and keep the input tensor list unchanged."""
    mps_in_change = mps.copy()
    mps_in_change[0] = mps_in_change[0] * num
    return mps_in_change


def mpo_mul_num(mpo, num):
    """Scale an MPO via its first core and keep the input tensor list unchanged."""
    mpo_ch = mpo.copy()
    mpo_ch[0] = mpo_ch[0] * num
    return mpo_ch


def mpo_multiplication(mpo_one, mpo_two):
    """Compose two MPOs using local operator contractions."""
    if len(mpo_one) != len(mpo_two):
        raise ValueError("Число тензоров должно совпадать")
    mult_mpo = []
    for a, b in zip(mpo_one, mpo_two):
        dtype = torch.promote_types(a.dtype, b.dtype)
        a, b = a.to(dtype=dtype), b.to(dtype=dtype)
        sh1, sh2 = a.shape, b.shape
        mult_mpo.append(torch.einsum('ijkl, mkop -> imjolp', a, b).reshape(
            sh1[0] * sh2[0], sh1[1], sh2[2], sh1[3] * sh2[3]))
    return mult_mpo


def mpo_x_mps(mpo, mps):
    """Apply an MPO to an MPS and return a list of output-state cores."""
    if len(mpo) != len(mps):
        raise ValueError("Число тензоров должно совпадать")
    new_mps = []
    for a, b in zip(mpo, mps):
        dtype = torch.promote_types(a.dtype, b.dtype)
        a, b = a.to(dtype=dtype), b.to(dtype=dtype)
        new_mps.append(torch.einsum("ijnl, mno -> imjlo", a, b).reshape(
            a.shape[0] * b.shape[0], a.shape[1], a.shape[-1] * b.shape[-1]))
    return new_mps


def mps_inner_product(psi_one, psi_two, conjugate_first=True):
    """Contract two tensor trains to a complex scalar.
    
    Parameters:
        psi_one: First input MPS (bra side of a scalar contraction).
        psi_two: Optional second input MPS (ket side of a contraction).
        conjugate_first: Whether to conjugate the first core sequence.
    
    Returns:
        Scalar torch tensor (with optional conjugation of the first MPS)."""
    if len(psi_one) != len(psi_two):
        raise ValueError("Число тензоров должно совпадать")
    if not psi_one:
        raise ValueError("Пустые MPS не поддерживаются")
    dtype = psi_one[0].dtype
    for t in [*psi_one, *psi_two]:
        dtype = torch.promote_types(dtype, t.dtype)
    env = torch.ones((1, 1), dtype=dtype, device=psi_one[0].device)
    for a, b in zip(psi_one, psi_two):
        a, b = a.to(dtype=dtype), b.to(dtype=dtype)
        if conjugate_first:
            a = a.conj()
        env = torch.einsum('ab,aic,bid->cd', env, a, b)
    return env.squeeze()


class MPS:
    """MPS: high-level wrapper around PyTorch tensor-train cores.
    
    Each wrapper stores a list of cores and exposes arithmetic
    and contraction operators; it does not automatically compress."""
    def __init__(self, tensors, is_bra=False):
        """Wrap a sequence of cores; does not clone their underlying torch tensors.
        
        Parameters:
            tensors: Sequence of PyTorch tensor cores.
            is_bra: For MPS, whether this object is oriented as a bra.
        
        Notes:
            MPS cores: (left_bond, physical, right_bond); MPO cores:
            (left_bond, physical_out, physical_in, right_bond)."""
        self.tensors = list(tensors)
        self.is_bra = is_bra

    def __len__(self):
        """Return the number of tensor-train sites."""
        return len(self.tensors)

    def __iter__(self):
        """Iterate over the stored tensor cores in their current site order."""
        return iter(self.tensors)

    def __getitem__(self, i):
        """Access a tensor core (or slice) through list indexing."""
        return self.tensors[i]

    def __setitem__(self, i, value):
        """Replace an indexed tensor core in this wrapper."""
        self.tensors[i] = value

    def copy(self):
        """Return a new wrapper with a shallow copy of the tensor list.
        
        Notes:
            The core tensors are shared with the original object; in-place edits affect both wrappers."""
        return MPS(self.tensors.copy(), self.is_bra)

    def clone(self):
        """Return a new wrapper with cloned values of all tensor cores.
        
        Notes:
            Core data are independently cloned, unlike copy()."""
        return MPS([t.clone() for t in self.tensors], self.is_bra)

    def __add__(self, other):
        """Add two compatible tensor trains without automatic bond compression."""
        if not isinstance(other, MPS):
            return NotImplemented
        if self.is_bra != other.is_bra:
            raise ValueError("Нельзя складывать бра и кет")
        return MPS(mps_sum(self.tensors, other.tensors), self.is_bra)

    def __neg__(self):
        """Return the additive inverse of this tensor train."""
        return self * (-1)

    def __sub__(self, other):
        """Subtract another compatible tensor train."""
        if not isinstance(other, MPS):
            return NotImplemented
        return self + (-other)

    def __mul__(self, num):
        """Scale the tensor train by a numeric scalar."""
        if not _is_number(num):
            return NotImplemented
        return MPS(mps_mul_num(self.tensors, num), self.is_bra)

    def __rmul__(self, num):
        """Implement scalar multiplication from the left."""
        return self * num

    def __truediv__(self, num):
        """Divide all amplitudes or matrix entries by a numeric scalar."""
        if not _is_number(num):
            return NotImplemented
        return self * (1 / num)

    def __matmul__(self, other):
        """Implement operator composition, action, scalar contractions or outer products.
        
        Parameters:
            other: Second tensor-train object, operator, or vector.
        
        Returns:
            MPS, MPO, scalar tensor, or NotImplemented depending on operands.
        
        Notes:
            MPS: ket @ bra produces an MPO; bra @ ket contracts; ket @ ket also contracts with conjugation of the first argument. MPO: @ composes operators or acts on a ket."""
        if isinstance(other, MPO):
            if not self.is_bra:
                return NotImplemented
            if len(self) != len(other):
                raise ValueError("Число тензоров должно совпадать")
            tensors = []
            for b, w in zip(self.tensors, other.tensors):
                dtype = torch.promote_types(b.dtype, w.dtype)
                b, w = b.to(dtype), w.to(dtype)
                tensors.append(torch.einsum('aic, bijd -> abjcd', b, w).reshape(
                    b.shape[0]*w.shape[0], w.shape[2], b.shape[2]*w.shape[3]))
            return MPS(tensors, is_bra=True)
        if not isinstance(other, MPS):
            return NotImplemented
        if not self.is_bra and other.is_bra:
            # psi @ phi.H -> |psi><phi| (MPO)
            if len(self) != len(other):
                raise ValueError("Число тензоров должно совпадать")
            tensors = []
            for a, b in zip(self.tensors, other.tensors):
                dtype = torch.promote_types(a.dtype, b.dtype)
                a, b = a.to(dtype), b.to(dtype)
                tensors.append(torch.einsum('aic, bjd -> abijcd', a, b).reshape(
                    a.shape[0]*b.shape[0], a.shape[1], b.shape[1], a.shape[2]*b.shape[2]))
            return MPO(tensors)
        if other.is_bra:
            raise ValueError("Два бра-вектора нельзя перемножить через @")
        # psi @ phi -> <psi|phi>; psi.T @ phi -> psi^T phi
        return mps_inner_product(self.tensors, other.tensors,
                                 conjugate_first=not self.is_bra)

    def conj(self):
        """Return a conjugated tensor train without changing bra/ket orientation."""
        return MPS([t.conj() for t in self.tensors], self.is_bra)

    def __and__(self, other):
        """Concatenate two tensor trains to represent a tensor product.
        
        Parameters:
            other: Second tensor-train object, operator, or vector.
        
        Notes:
            The connecting boundary bonds must be one for a valid tensor-product concatenation. No dimension validation is performed here."""
        if not isinstance(other, MPS):
            return NotImplemented
        if self.is_bra != other.is_bra:
            raise ValueError("Нельзя объединять бра и кет")

        return MPS(self.tensors + other.tensors, self.is_bra)

    @property
    def T(self):
        """Toggle ket/bra orientation (MPS) or transpose physical MPO indices.
        
        Notes:
            For MPS, this toggles is_bra without numerical conjugation. For MPO, it swaps physical indices."""
        return MPS(self.tensors, not self.is_bra)

    @property
    def H(self):
        """Apply Hermitian conjugation to the MPS or MPO.
        
        Notes:
            For MPS, this conjugates cores and toggles is_bra. For MPO, it also swaps input/output physical indices."""
        return MPS([t.conj() for t in self.tensors], not self.is_bra)


class MPO:
    """MPO: high-level wrapper around PyTorch tensor-train cores.
    
    Each wrapper stores a list of cores and exposes arithmetic
    and contraction operators; it does not automatically compress."""
    def __init__(self, tensors):
        """Wrap a sequence of cores; does not clone their underlying torch tensors.
        
        Parameters:
            tensors: Sequence of PyTorch tensor cores.
        
        Notes:
            MPS cores: (left_bond, physical, right_bond); MPO cores:
            (left_bond, physical_out, physical_in, right_bond)."""
        self.tensors = list(tensors)

    def __len__(self):
        """Return the number of tensor-train sites."""
        return len(self.tensors)

    def __iter__(self):
        """Iterate over the stored tensor cores in their current site order."""
        return iter(self.tensors)

    def __getitem__(self, i):
        """Access a tensor core (or slice) through list indexing."""
        return self.tensors[i]

    def __setitem__(self, i, value):
        """Replace an indexed tensor core in this wrapper."""
        self.tensors[i] = value

    def copy(self):
        """Return a new wrapper with a shallow copy of the tensor list.
        
        Notes:
            The core tensors are shared with the original object; in-place edits affect both wrappers."""
        return MPO(self.tensors.copy())

    def clone(self):
        """Return a new wrapper with cloned values of all tensor cores.
        
        Notes:
            Core data are independently cloned, unlike copy()."""
        return MPO([t.clone() for t in self.tensors])

    def __add__(self, other):
        """Add two compatible tensor trains without automatic bond compression."""
        if not isinstance(other, MPO):
            return NotImplemented
        return MPO(mpo_sum(self.tensors, other.tensors))

    def __neg__(self):
        """Return the additive inverse of this tensor train."""
        return self * (-1)

    def __sub__(self, other):
        """Subtract another compatible tensor train."""
        if not isinstance(other, MPO):
            return NotImplemented
        return self + (-other)

    def __mul__(self, num):
        """Scale the tensor train by a numeric scalar."""
        if not _is_number(num):
            return NotImplemented
        return MPO(mpo_mul_num(self.tensors, num))

    def __rmul__(self, num):
        """Implement scalar multiplication from the left."""
        return self * num

    def __truediv__(self, num):
        """Divide all amplitudes or matrix entries by a numeric scalar."""
        if not _is_number(num):
            return NotImplemented
        return self * (1 / num)

    def __matmul__(self, other):
        """Implement operator composition, action, scalar contractions or outer products.
        
        Parameters:
            other: Second tensor-train object, operator, or vector.
        
        Returns:
            MPS, MPO, scalar tensor, or NotImplemented depending on operands.
        
        Notes:
            MPS: ket @ bra produces an MPO; bra @ ket contracts; ket @ ket also contracts with conjugation of the first argument. MPO: @ composes operators or acts on a ket."""
        if isinstance(other, MPO):
            return MPO(mpo_multiplication(self.tensors, other.tensors))
        if isinstance(other, MPS):
            if other.is_bra:
                raise ValueError("MPO может действовать справа только на MPS-кет")
            return MPS(mpo_x_mps(self.tensors, other.tensors))
        return NotImplemented

    def conj(self):
        """Return a conjugated tensor train without changing bra/ket orientation."""
        return MPO([t.conj() for t in self.tensors])

    def __and__(self, other):
        """Concatenate two tensor trains to represent a tensor product.
        
        Parameters:
            other: Second tensor-train object, operator, or vector.
        
        Notes:
            The connecting boundary bonds must be one for a valid tensor-product concatenation. No dimension validation is performed here."""
        if not isinstance(other, MPO):
            return NotImplemented

        return MPO(self.tensors + other.tensors)

    @property
    def T(self):
        """Toggle ket/bra orientation (MPS) or transpose physical MPO indices.
        
        Notes:
            For MPS, this toggles is_bra without numerical conjugation. For MPO, it swaps physical indices."""
        return MPO([t.transpose(1, 2) for t in self.tensors])

    @property
    def H(self):
        """Apply Hermitian conjugation to the MPS or MPO.
        
        Notes:
            For MPS, this conjugates cores and toggles is_bra. For MPO, it also swaps input/output physical indices."""
        return MPO([t.transpose(1, 2).conj() for t in self.tensors])
