Help on Tensor in module torch object:

class TTeennssoorr(torch._C._TensorBase)
 |  Method resolution order:
 |      Tensor
 |      torch._C._TensorBase
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ____aabbss____ = abs(...)
 |  
 |  ____aarrrraayy____(self, dtype=None)
 |  
 |  ____aarrrraayy__wwrraapp____(self, array)
 |      # Wrap Numpy array again in a suitable tensor when done, to support e.g.
 |      # `numpy.sin(tensor) -> tensor` or `numpy.greater(tensor, 0) -> ByteTensor`
 |  
 |  ____ccoonnttaaiinnss____(self, element)
 |      Check if `element` is present in tensor
 |      
 |      Args:
 |          element (Tensor or scalar): element to be checked
 |              for presence in current tensor"
 |  
 |  ____ddeeeeppccooppyy____(self, memo)
 |  
 |  ____ddiirr____(self)
 |      Default dir() implementation.
 |  
 |  ____fflloooorrddiivv____(self, other)
 |  
 |  ____ffoorrmmaatt____(self, format_spec)
 |      Default object formatter.
 |  
 |  ____hhaasshh____(self)
 |      Return hash(self).
 |  
 |  ____iippooww____(self, other)
 |  
 |  ____iitteerr____(self)
 |  
 |  ____iittrruueeddiivv____ = __idiv__(...)
 |  
 |  ____lleenn____(self)
 |      Return len(self).
 |  
 |  ____nneegg____ = neg(...)
 |  
 |  ____ppooww____ = pow(...)
 |  
 |  ____rrddiivv____(self, other)
 |  
 |  ____rreedduuccee__eexx____(self, proto)
 |      Helper for pickle.
 |  
 |  ____rreepprr____(self)
 |      Return repr(self).
 |  
 |  ____rreevveerrsseedd____(self)
 |      Reverses the tensor along dimension 0.
 |  
 |  ____rrfflloooorrddiivv____(self, other)
 |  
 |  ____rrppooww____(self, other)
 |  
 |  ____rrssuubb____(self, other)
 |  
 |  ____rrttrruueeddiivv____ = __rdiv__(self, other)
 |  
 |  ____sseettssttaattee____(self, state)
 |  
 |  aalliiggnn__ttoo(self, *names)
 |      Permutes the dimensions of the :attr:`self` tensor to match the order
 |      specified in :attr:`names`, adding size-one dims for any new names.
 |      
 |      All of the dims of :attr:`self` must be named in order to use this method.
 |      The resulting tensor is a view on the original tensor.
 |      
 |      All dimension names of :attr:`self` must be present in :attr:`names`.
 |      :attr:`names` may contain additional names that are not in ``self.names``;
 |      the output tensor has a size-one dimension for each of those new names.
 |      
 |      :attr:`names` may contain up to one Ellipsis (``...``).
 |      The Ellipsis is expanded to be equal to all dimension names of :attr:`self`
 |      that are not mentioned in :attr:`names`, in the order that they appear
 |      in :attr:`self`.
 |      
 |      Python 2 does not support Ellipsis but one may use a string literal
 |      instead (``'...'``).
 |      
 |      Args:
 |          names (iterable of str): The desired dimension ordering of the
 |              output tensor. May contain up to one Ellipsis that is expanded
 |              to all unmentioned dim names of :attr:`self`.
 |      
 |      Examples::
 |      
 |          >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
 |          >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')
 |      
 |          # Move the F and E dims to the front while keeping the rest in order
 |          >>> named_tensor.align_to('F', 'E', ...)
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  bbaacckkwwaarrdd(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)
 |      Computes the gradient of current tensor w.r.t. graph leaves.
 |      
 |      The graph is differentiated using the chain rule. If the tensor is
 |      non-scalar (i.e. its data has more than one element) and requires
 |      gradient, the function additionally requires specifying ``gradient``.
 |      It should be a tensor of matching type and location, that contains
 |      the gradient of the differentiated function w.r.t. ``self``.
 |      
 |      This function accumulates gradients in the leaves - you might need to zero
 |      ``.grad`` attributes or set them to ``None`` before calling it.
 |      See :ref:`Default gradient layouts<default-grad-layouts>`
 |      for details on the memory layout of accumulated gradients.
 |      
 |      .. note::
 |      
 |          If you run any forward ops, create ``gradient``, and/or call ``backward``
 |          in a user-specified CUDA stream context, see
 |          :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.
 |      
 |      Args:
 |          gradient (Tensor or None): Gradient w.r.t. the
 |              tensor. If it is a tensor, it will be automatically converted
 |              to a Tensor that does not require grad unless ``create_graph`` is True.
 |              None values can be specified for scalar Tensors or ones that
 |              don't require grad. If a None value would be acceptable then
 |              this argument is optional.
 |          retain_graph (bool, optional): If ``False``, the graph used to compute
 |              the grads will be freed. Note that in nearly all cases setting
 |              this option to True is not needed and often can be worked around
 |              in a much more efficient way. Defaults to the value of
 |              ``create_graph``.
 |          create_graph (bool, optional): If ``True``, graph of the derivative will
 |              be constructed, allowing to compute higher order derivative
 |              products. Defaults to ``False``.
 |          inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
 |              accumulated into ``.grad``. All other Tensors will be ignored. If not
 |              provided, the gradient is accumulated into all the leaf Tensors that were
 |              used to compute the attr::tensors. All the provided inputs must be leaf
 |              Tensors.
 |  
 |  ddeettaacchh(...)
 |      Returns a new Tensor, detached from the current graph.
 |      
 |      The result will never require gradient.
 |      
 |      .. note::
 |      
 |        Returned Tensor shares the same storage with the original one.
 |        In-place modifications on either of them will be seen, and may trigger
 |        errors in correctness checks.
 |        IMPORTANT NOTE: Previously, in-place size / stride / storage changes
 |        (such as `resize_` / `resize_as_` / `set_` / `transpose_`) to the returned tensor
 |        also update the original tensor. Now, these in-place changes will not update the
 |        original tensor anymore, and will instead trigger an error.
 |        For sparse tensors:
 |        In-place indices / values changes (such as `zero_` / `copy_` / `add_`) to the
 |        returned tensor will not update the original tensor anymore, and will instead
 |        trigger an error.
 |  
 |  ddeettaacchh__(...)
 |      Detaches the Tensor from the graph that created it, making it a leaf.
 |      Views cannot be detached in-place.
 |  
 |  iiss__sshhaarreedd(self)
 |      Checks if tensor is in shared memory.
 |      
 |      This is always ``True`` for CUDA tensors.
 |  
 |  iissttfftt(self, n_fft: int, hop_length: Union[int, NoneType] = None, win_length: Union[int, NoneType] = None, window: 'Optional[Tensor]' = None, center: bool = True, normalized: bool = False, onesided: Union[bool, NoneType] = None, length: Union[int, NoneType] = None, return_complex: bool = False)
 |      See :func:`torch.istft`
 |  
 |  lluu(self, pivot=True, get_infos=False)
 |      See :func:`torch.lu`
 |  
 |  nnoorrmm(self, p='fro', dim=None, keepdim=False, dtype=None)
 |      See :func:`torch.norm`
 |  
 |  rreeffiinnee__nnaammeess(self, *names)
 |      Refines the dimension names of :attr:`self` according to :attr:`names`.
 |      
 |      Refining is a special case of renaming that "lifts" unnamed dimensions.
 |      A ``None`` dim can be refined to have any name; a named dim can only be
 |      refined to have the same name.
 |      
 |      Because named tensors can coexist with unnamed tensors, refining names
 |      gives a nice way to write named-tensor-aware code that works with both
 |      named and unnamed tensors.
 |      
 |      :attr:`names` may contain up to one Ellipsis (``...``).
 |      The Ellipsis is expanded greedily; it is expanded in-place to fill
 |      :attr:`names` to the same length as ``self.dim()`` using names from the
 |      corresponding indices of ``self.names``.
 |      
 |      Python 2 does not support Ellipsis but one may use a string literal
 |      instead (``'...'``).
 |      
 |      Args:
 |          names (iterable of str): The desired names of the output tensor. May
 |              contain up to one Ellipsis.
 |      
 |      Examples::
 |      
 |          >>> imgs = torch.randn(32, 3, 128, 128)
 |          >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
 |          >>> named_imgs.names
 |          ('N', 'C', 'H', 'W')
 |      
 |          >>> tensor = torch.randn(2, 3, 5, 7, 11)
 |          >>> tensor = tensor.refine_names('A', ..., 'B', 'C')
 |          >>> tensor.names
 |          ('A', None, None, 'B', 'C')
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  rreeggiisstteerr__hhooookk(self, hook)
 |      Registers a backward hook.
 |      
 |      The hook will be called every time a gradient with respect to the
 |      Tensor is computed. The hook should have the following signature::
 |      
 |          hook(grad) -> Tensor or None
 |      
 |      
 |      The hook should not modify its argument, but it can optionally return
 |      a new gradient which will be used in place of :attr:`grad`.
 |      
 |      This function returns a handle with a method ``handle.remove()``
 |      that removes the hook from the module.
 |      
 |      Example::
 |      
 |          >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
 |          >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
 |          >>> v.backward(torch.tensor([1., 2., 3.]))
 |          >>> v.grad
 |      
 |           2
 |           4
 |           6
 |          [torch.FloatTensor of size (3,)]
 |      
 |          >>> h.remove()  # removes the hook
 |  
 |  rreeiinnffoorrccee(self, reward)
 |  
 |  rreennaammee(self, *names, **rename_map)
 |      Renames dimension names of :attr:`self`.
 |      
 |      There are two main usages:
 |      
 |      ``self.rename(**rename_map)`` returns a view on tensor that has dims
 |      renamed as specified in the mapping :attr:`rename_map`.
 |      
 |      ``self.rename(*names)`` returns a view on tensor, renaming all
 |      dimensions positionally using :attr:`names`.
 |      Use ``self.rename(None)`` to drop names on a tensor.
 |      
 |      One cannot specify both positional args :attr:`names` and keyword args
 |      :attr:`rename_map`.
 |      
 |      Examples::
 |      
 |          >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
 |          >>> renamed_imgs = imgs.rename(N='batch', C='channels')
 |          >>> renamed_imgs.names
 |          ('batch', 'channels', 'H', 'W')
 |      
 |          >>> renamed_imgs = imgs.rename(None)
 |          >>> renamed_imgs.names
 |          (None,)
 |      
 |          >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
 |          >>> renamed_imgs.names
 |          ('batch', 'channel', 'height', 'width')
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  rreennaammee__(self, *names, **rename_map)
 |      In-place version of :meth:`~Tensor.rename`.
 |  
 |  rreessiizzee(self, *sizes)
 |  
 |  rreessiizzee__aass(self, tensor)
 |  
 |  rreettaaiinn__ggrraadd(self)
 |      Enables .grad attribute for non-leaf Tensors.
 |  
 |  sshhaarree__mmeemmoorryy__(self)
 |      Moves the underlying storage to shared memory.
 |      
 |      This is a no-op if the underlying storage is already in shared memory
 |      and for CUDA tensors. Tensors in shared memory cannot be resized.
 |  
 |  sspplliitt(self, split_size, dim=0)
 |      See :func:`torch.split`
 |  
 |  ssttfftt(self, n_fft: int, hop_length: Union[int, NoneType] = None, win_length: Union[int, NoneType] = None, window: 'Optional[Tensor]' = None, center: bool = True, pad_mode: str = 'reflect', normalized: bool = False, onesided: Union[bool, NoneType] = None, return_complex: Union[bool, NoneType] = None)
 |      See :func:`torch.stft`
 |      
 |      .. warning::
 |        This function changed signature at version 0.4.1. Calling with
 |        the previous signature may cause error or return incorrect result.
 |  
 |  uunnffllaatttteenn(self, dim, sizes)
 |      Expands the dimension :attr:`dim` of the :attr:`self` tensor over multiple dimensions
 |      of sizes given by :attr:`sizes`.
 |      
 |      * :attr:`sizes` is the new shape of the unflattened dimension and it can be a `Tuple[int]` as well
 |        as `torch.Size` if :attr:`self` is a `Tensor`, or `namedshape` (Tuple[(name: str, size: int)])
 |        if :attr:`self` is a `NamedTensor`. The total number of elements in sizes must match the number
 |        of elements in the original dim being unflattened.
 |      
 |      Args:
 |          dim (Union[int, str]): Dimension to unflatten
 |          sizes (Union[Tuple[int] or torch.Size, Tuple[Tuple[str, int]]]): New shape of the unflattened dimension
 |      
 |      Examples:
 |          >>> torch.randn(3, 4, 1).unflatten(1, (2, 2)).shape
 |          torch.Size([3, 2, 2, 1])
 |          >>> torch.randn(2, 4, names=('A', 'B')).unflatten('B', (('B1', 2), ('B2', 2)))
 |          tensor([[[-1.1772,  0.0180],
 |                  [ 0.2412,  0.1431]],
 |      
 |                  [[-1.1819, -0.8899],
 |                  [ 1.5813,  0.2274]]], names=('A', 'B1', 'B2'))
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  uunniiqquuee(self, sorted=True, return_inverse=False, return_counts=False, dim=None)
 |      Returns the unique elements of the input tensor.
 |      
 |      See :func:`torch.unique`
 |  
 |  uunniiqquuee__ccoonnsseeccuuttiivvee(self, return_inverse=False, return_counts=False, dim=None)
 |      Eliminates all but the first element from every consecutive group of equivalent elements.
 |      
 |      See :func:`torch.unique_consecutive`
 |  
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |  
 |  ____ttoorrcchh__ffuunnccttiioonn____(func, types, args=(), kwargs=None) from builtins.type
 |      This __torch_function__ implementation wraps subclasses such that
 |      methods called on subclasses return a subclass instance instead of
 |      a ``torch.Tensor`` instance.
 |      
 |      One corollary to this is that you need coverage for torch.Tensor
 |      methods if implementing __torch_function__ for subclasses.
 |      
 |      We recommend always calling ``super().__torch_function__`` as the base
 |      case when doing the above.
 |      
 |      While not mandatory, we recommend making `__torch_function__` a classmethod.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |  
 |  ____ccuuddaa__aarrrraayy__iinntteerrffaaccee____
 |      Array view description for cuda tensors.
 |      
 |      See:
 |      https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  ____ddiicctt____
 |      dictionary for instance variables (if defined)
 |  
 |  ____wweeaakkrreeff____
 |      list of weak references to the object (if defined)
 |  
 |  ggrraadd
 |      This attribute is ``None`` by default and becomes a Tensor the first time a call to
 |      :func:`backward` computes gradients for ``self``.
 |      The attribute will then contain the gradients computed and future calls to
 |      :func:`backward` will accumulate (add) gradients into it.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  ____aarrrraayy__pprriioorriittyy____ = 1000
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from torch._C._TensorBase:
 |  
 |  ____aadddd____(...)
 |  
 |  ____aanndd____(...)
 |  
 |  ____bbooooll____(...)
 |  
 |  ____ccoommpplleexx____(...)
 |  
 |  ____ddeelliitteemm____(self, key, /)
 |      Delete self[key].
 |  
 |  ____ddiivv____(...)
 |  
 |  ____eeqq____(...)
 |      Return self==value.
 |  
 |  ____ffllooaatt____(...)
 |  
 |  ____ggee____(...)
 |      Return self>=value.
 |  
 |  ____ggeettiitteemm____(self, key, /)
 |      Return self[key].
 |  
 |  ____ggtt____(...)
 |      Return self>value.
 |  
 |  ____iiaadddd____(...)
 |  
 |  ____iiaanndd____(...)
 |  
 |  ____iiddiivv____(...)
 |  
 |  ____iifflloooorrddiivv____(...)
 |  
 |  ____iillsshhiifftt____(...)
 |  
 |  ____iimmoodd____(...)
 |  
 |  ____iimmuull____(...)
 |  
 |  ____iinnddeexx____(...)
 |  
 |  ____iinntt____(...)
 |  
 |  ____iinnvveerrtt____(...)
 |  
 |  ____iioorr____(...)
 |  
 |  ____iirrsshhiifftt____(...)
 |  
 |  ____iissuubb____(...)
 |  
 |  ____iixxoorr____(...)
 |  
 |  ____llee____(...)
 |      Return self<=value.
 |  
 |  ____lloonngg____(...)
 |  
 |  ____llsshhiifftt____(...)
 |  
 |  ____lltt____(...)
 |      Return self<value.
 |  
 |  ____mmaattmmuull____(...)
 |  
 |  ____mmoodd____(...)
 |  
 |  ____mmuull____(...)
 |  
 |  ____nnee____(...)
 |      Return self!=value.
 |  
 |  ____nnoonnzzeerroo____(...)
 |  
 |  ____oorr____(...)
 |  
 |  ____rraadddd____(...)
 |  
 |  ____rrmmuull____(...)
 |  
 |  ____rrsshhiifftt____(...)
 |  
 |  ____sseettiitteemm____(self, key, value, /)
 |      Set self[key] to value.
 |  
 |  ____ssuubb____(...)
 |  
 |  ____ttrruueeddiivv____(...)
 |  
 |  ____xxoorr____(...)
 |  
 |  aabbss(...)
 |      abs() -> Tensor
 |      
 |      See :func:`torch.abs`
 |  
 |  aabbss__(...)
 |      abs_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.abs`
 |  
 |  aabbssoolluuttee(...)
 |      absolute() -> Tensor
 |      
 |      Alias for :func:`abs`
 |  
 |  aabbssoolluuttee__(...)
 |      absolute_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.absolute`
 |      Alias for :func:`abs_`
 |  
 |  aaccooss(...)
 |      acos() -> Tensor
 |      
 |      See :func:`torch.acos`
 |  
 |  aaccooss__(...)
 |      acos_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.acos`
 |  
 |  aaccoosshh(...)
 |      acosh() -> Tensor
 |      
 |      See :func:`torch.acosh`
 |  
 |  aaccoosshh__(...)
 |      acosh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.acosh`
 |  
 |  aadddd(...)
 |      add(other, *, alpha=1) -> Tensor
 |      
 |      Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
 |      and :attr:`other` are specified, each element of :attr:`other` is scaled by
 |      :attr:`alpha` before being used.
 |      
 |      When :attr:`other` is a tensor, the shape of :attr:`other` must be
 |      :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
 |      tensor
 |      
 |      See :func:`torch.add`
 |  
 |  aadddd__(...)
 |      add_(other, *, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.add`
 |  
 |  aaddddbbmmmm(...)
 |      addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.addbmm`
 |  
 |  aaddddbbmmmm__(...)
 |      addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addbmm`
 |  
 |  aaddddccddiivv(...)
 |      addcdiv(tensor1, tensor2, *, value=1) -> Tensor
 |      
 |      See :func:`torch.addcdiv`
 |  
 |  aaddddccddiivv__(...)
 |      addcdiv_(tensor1, tensor2, *, value=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addcdiv`
 |  
 |  aaddddccmmuull(...)
 |      addcmul(tensor1, tensor2, *, value=1) -> Tensor
 |      
 |      See :func:`torch.addcmul`
 |  
 |  aaddddccmmuull__(...)
 |      addcmul_(tensor1, tensor2, *, value=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addcmul`
 |  
 |  aaddddmmmm(...)
 |      addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.addmm`
 |  
 |  aaddddmmmm__(...)
 |      addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addmm`
 |  
 |  aaddddmmvv(...)
 |      addmv(mat, vec, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.addmv`
 |  
 |  aaddddmmvv__(...)
 |      addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addmv`
 |  
 |  aaddddrr(...)
 |      addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.addr`
 |  
 |  aaddddrr__(...)
 |      addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.addr`
 |  
 |  aalliiggnn__aass(...)
 |      align_as(other) -> Tensor
 |      
 |      Permutes the dimensions of the :attr:`self` tensor to match the dimension order
 |      in the :attr:`other` tensor, adding size-one dims for any new names.
 |      
 |      This operation is useful for explicit broadcasting by names (see examples).
 |      
 |      All of the dims of :attr:`self` must be named in order to use this method.
 |      The resulting tensor is a view on the original tensor.
 |      
 |      All dimension names of :attr:`self` must be present in ``other.names``.
 |      :attr:`other` may contain named dimensions that are not in ``self.names``;
 |      the output tensor has a size-one dimension for each of those new names.
 |      
 |      To align a tensor to a specific order, use :meth:`~Tensor.align_to`.
 |      
 |      Examples::
 |      
 |          # Example 1: Applying a mask
 |          >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
 |          >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
 |          >>> imgs.masked_fill_(mask.align_as(imgs), 0)
 |      
 |      
 |          # Example 2: Applying a per-channel-scale
 |          >>> def scale_channels(input, scale):
 |          >>>    scale = scale.refine_names('C')
 |          >>>    return input * scale.align_as(input)
 |      
 |          >>> num_channels = 3
 |          >>> scale = torch.randn(num_channels, names=('C',))
 |          >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
 |          >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
 |          >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))
 |      
 |          # scale_channels is agnostic to the dimension order of the input
 |          >>> scale_channels(imgs, scale)
 |          >>> scale_channels(more_imgs, scale)
 |          >>> scale_channels(videos, scale)
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  aallll(...)
 |      all(dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.all`
 |  
 |  aallllcclloossee(...)
 |      allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
 |      
 |      See :func:`torch.allclose`
 |  
 |  aammaaxx(...)
 |      amax(dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.amax`
 |  
 |  aammiinn(...)
 |      amin(dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.amin`
 |  
 |  aannggllee(...)
 |      angle() -> Tensor
 |      
 |      See :func:`torch.angle`
 |  
 |  aannyy(...)
 |      any(dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.any`
 |  
 |  aappppllyy__(...)
 |      apply_(callable) -> Tensor
 |      
 |      Applies the function :attr:`callable` to each element in the tensor, replacing
 |      each element with the value returned by :attr:`callable`.
 |      
 |      .. note::
 |      
 |          This function only works with CPU tensors and should not be used in code
 |          sections that require high performance.
 |  
 |  aarrccccooss(...)
 |      arccos() -> Tensor
 |      
 |      See :func:`torch.arccos`
 |  
 |  aarrccccooss__(...)
 |      arccos_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arccos`
 |  
 |  aarrccccoosshh(...)
 |      acosh() -> Tensor
 |      
 |      See :func:`torch.arccosh`
 |  
 |  aarrccccoosshh__(...)
 |      acosh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arccosh`
 |  
 |  aarrccssiinn(...)
 |      arcsin() -> Tensor
 |      
 |      See :func:`torch.arcsin`
 |  
 |  aarrccssiinn__(...)
 |      arcsin_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arcsin`
 |  
 |  aarrccssiinnhh(...)
 |      arcsinh() -> Tensor
 |      
 |      See :func:`torch.arcsinh`
 |  
 |  aarrccssiinnhh__(...)
 |      arcsinh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arcsinh`
 |  
 |  aarrccttaann(...)
 |      arctan() -> Tensor
 |      
 |      See :func:`torch.arctan`
 |  
 |  aarrccttaann__(...)
 |      arctan_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arctan`
 |  
 |  aarrccttaannhh(...)
 |      arctanh() -> Tensor
 |      
 |      See :func:`torch.arctanh`
 |  
 |  aarrccttaannhh__(...)
 |      arctanh_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.arctanh`
 |  
 |  aarrggmmaaxx(...)
 |      argmax(dim=None, keepdim=False) -> LongTensor
 |      
 |      See :func:`torch.argmax`
 |  
 |  aarrggmmiinn(...)
 |      argmin(dim=None, keepdim=False) -> LongTensor
 |      
 |      See :func:`torch.argmin`
 |  
 |  aarrggssoorrtt(...)
 |      argsort(dim=-1, descending=False) -> LongTensor
 |      
 |      See :func:`torch.argsort`
 |  
 |  aass__ssttrriiddeedd(...)
 |      as_strided(size, stride, storage_offset=0) -> Tensor
 |      
 |      See :func:`torch.as_strided`
 |  
 |  aass__ssttrriiddeedd__(...)
 |  
 |  aass__ssuubbccllaassss(...)
 |      as_subclass(cls) -> Tensor
 |      
 |      Makes a ``cls`` instance with the same data pointer as ``self``. Changes
 |      in the output mirror changes in ``self``, and the output stays attached
 |      to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
 |  
 |  aassiinn(...)
 |      asin() -> Tensor
 |      
 |      See :func:`torch.asin`
 |  
 |  aassiinn__(...)
 |      asin_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.asin`
 |  
 |  aassiinnhh(...)
 |      asinh() -> Tensor
 |      
 |      See :func:`torch.asinh`
 |  
 |  aassiinnhh__(...)
 |      asinh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.asinh`
 |  
 |  aattaann(...)
 |      atan() -> Tensor
 |      
 |      See :func:`torch.atan`
 |  
 |  aattaann22(...)
 |      atan2(other) -> Tensor
 |      
 |      See :func:`torch.atan2`
 |  
 |  aattaann22__(...)
 |      atan2_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.atan2`
 |  
 |  aattaann__(...)
 |      atan_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.atan`
 |  
 |  aattaannhh(...)
 |      atanh() -> Tensor
 |      
 |      See :func:`torch.atanh`
 |  
 |  aattaannhh__(...)
 |      atanh_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.atanh`
 |  
 |  bbaaddddbbmmmm(...)
 |      baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.baddbmm`
 |  
 |  bbaaddddbbmmmm__(...)
 |      baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.baddbmm`
 |  
 |  bbeerrnnoouullllii(...)
 |      bernoulli(*, generator=None) -> Tensor
 |      
 |      Returns a result tensor where each :math:`\texttt{result[i]}` is independently
 |      sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
 |      floating point ``dtype``, and the result will have the same ``dtype``.
 |      
 |      See :func:`torch.bernoulli`
 |  
 |  bbeerrnnoouullllii__(...)
 |      .. function:: bernoulli_(p=0.5, *, generator=None) -> Tensor
 |      
 |          Fills each location of :attr:`self` with an independent sample from
 |          :math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
 |          ``dtype``.
 |      
 |      .. function:: bernoulli_(p_tensor, *, generator=None) -> Tensor
 |      
 |          :attr:`p_tensor` should be a tensor containing probabilities to be used for
 |          drawing the binary random number.
 |      
 |          The :math:`\text{i}^{th}` element of :attr:`self` tensor will be set to a
 |          value sampled from :math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`.
 |      
 |          :attr:`self` can have integral ``dtype``, but :attr:`p_tensor` must have
 |          floating point ``dtype``.
 |      
 |      See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
 |  
 |  bbffllooaatt1166(...)
 |      bfloat16(memory_format=torch.preserve_format) -> Tensor
 |      ``self.bfloat16()`` is equivalent to ``self.to(torch.bfloat16)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  bbiinnccoouunntt(...)
 |      bincount(weights=None, minlength=0) -> Tensor
 |      
 |      See :func:`torch.bincount`
 |  
 |  bbiittwwiissee__aanndd(...)
 |      bitwise_and() -> Tensor
 |      
 |      See :func:`torch.bitwise_and`
 |  
 |  bbiittwwiissee__aanndd__(...)
 |      bitwise_and_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.bitwise_and`
 |  
 |  bbiittwwiissee__nnoott(...)
 |      bitwise_not() -> Tensor
 |      
 |      See :func:`torch.bitwise_not`
 |  
 |  bbiittwwiissee__nnoott__(...)
 |      bitwise_not_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.bitwise_not`
 |  
 |  bbiittwwiissee__oorr(...)
 |      bitwise_or() -> Tensor
 |      
 |      See :func:`torch.bitwise_or`
 |  
 |  bbiittwwiissee__oorr__(...)
 |      bitwise_or_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.bitwise_or`
 |  
 |  bbiittwwiissee__xxoorr(...)
 |      bitwise_xor() -> Tensor
 |      
 |      See :func:`torch.bitwise_xor`
 |  
 |  bbiittwwiissee__xxoorr__(...)
 |      bitwise_xor_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.bitwise_xor`
 |  
 |  bbmmmm(...)
 |      bmm(batch2) -> Tensor
 |      
 |      See :func:`torch.bmm`
 |  
 |  bbooooll(...)
 |      bool(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.bool()`` is equivalent to ``self.to(torch.bool)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  bbrrooaaddccaasstt__ttoo(...)
 |      broadcast_to(shape) -> Tensor
 |      
 |      See :func:`torch.broadcast_to`.
 |  
 |  bbyyttee(...)
 |      byte(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  ccaauucchhyy__(...)
 |      cauchy_(median=0, sigma=1, *, generator=None) -> Tensor
 |      
 |      Fills the tensor with numbers drawn from the Cauchy distribution:
 |      
 |      .. math::
 |      
 |          f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}
 |  
 |  cceeiill(...)
 |      ceil() -> Tensor
 |      
 |      See :func:`torch.ceil`
 |  
 |  cceeiill__(...)
 |      ceil_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.ceil`
 |  
 |  cchhaarr(...)
 |      char(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  cchhoolleesskkyy(...)
 |      cholesky(upper=False) -> Tensor
 |      
 |      See :func:`torch.cholesky`
 |  
 |  cchhoolleesskkyy__iinnvveerrssee(...)
 |      cholesky_inverse(upper=False) -> Tensor
 |      
 |      See :func:`torch.cholesky_inverse`
 |  
 |  cchhoolleesskkyy__ssoollvvee(...)
 |      cholesky_solve(input2, upper=False) -> Tensor
 |      
 |      See :func:`torch.cholesky_solve`
 |  
 |  cchhuunnkk(...)
 |      chunk(chunks, dim=0) -> List of Tensors
 |      
 |      See :func:`torch.chunk`
 |  
 |  ccllaammpp(...)
 |      clamp(min, max) -> Tensor
 |      
 |      See :func:`torch.clamp`
 |  
 |  ccllaammpp__(...)
 |      clamp_(min, max) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.clamp`
 |  
 |  ccllaammpp__mmaaxx(...)
 |  
 |  ccllaammpp__mmaaxx__(...)
 |  
 |  ccllaammpp__mmiinn(...)
 |  
 |  ccllaammpp__mmiinn__(...)
 |  
 |  cclliipp(...)
 |      clip(min, max) -> Tensor
 |      
 |      Alias for :meth:`~Tensor.clamp`.
 |  
 |  cclliipp__(...)
 |      clip_(min, max) -> Tensor
 |      
 |      Alias for :meth:`~Tensor.clamp_`.
 |  
 |  cclloonnee(...)
 |      clone(*, memory_format=torch.preserve_format) -> Tensor
 |      
 |      See :func:`torch.clone`
 |  
 |  ccooaalleessccee(...)
 |      coalesce() -> Tensor
 |      
 |      Returns a coalesced copy of :attr:`self` if :attr:`self` is an
 |      :ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.
 |      
 |      Returns :attr:`self` if :attr:`self` is a coalesced tensor.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse COO tensor.
 |  
 |  ccoonnjj(...)
 |      conj() -> Tensor
 |      
 |      See :func:`torch.conj`
 |  
 |  ccoonnttiigguuoouuss(...)
 |      contiguous(memory_format=torch.contiguous_format) -> Tensor
 |      
 |      Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
 |      :attr:`self` tensor is already in the specified memory format, this function returns the
 |      :attr:`self` tensor.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.contiguous_format``.
 |  
 |  ccooppyy__(...)
 |      copy_(src, non_blocking=False) -> Tensor
 |      
 |      Copies the elements from :attr:`src` into :attr:`self` tensor and returns
 |      :attr:`self`.
 |      
 |      The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
 |      with the :attr:`self` tensor. It may be of a different data type or reside on a
 |      different device.
 |      
 |      Args:
 |          src (Tensor): the source tensor to copy from
 |          non_blocking (bool): if ``True`` and this copy is between CPU and GPU,
 |              the copy may occur asynchronously with respect to the host. For other
 |              cases, this argument has no effect.
 |  
 |  ccooppyyssiiggnn(...)
 |      copysign(other) -> Tensor
 |      
 |      See :func:`torch.copysign`
 |  
 |  ccooppyyssiiggnn__(...)
 |      copysign_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.copysign`
 |  
 |  ccooss(...)
 |      cos() -> Tensor
 |      
 |      See :func:`torch.cos`
 |  
 |  ccooss__(...)
 |      cos_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.cos`
 |  
 |  ccoosshh(...)
 |      cosh() -> Tensor
 |      
 |      See :func:`torch.cosh`
 |  
 |  ccoosshh__(...)
 |      cosh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.cosh`
 |  
 |  ccoouunntt__nnoonnzzeerroo(...)
 |      count_nonzero(dim=None) -> Tensor
 |      
 |      See :func:`torch.count_nonzero`
 |  
 |  ccppuu(...)
 |      cpu(memory_format=torch.preserve_format) -> Tensor
 |      
 |      Returns a copy of this object in CPU memory.
 |      
 |      If this object is already in CPU memory and on the correct device,
 |      then no copy is performed and the original object is returned.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  ccrroossss(...)
 |      cross(other, dim=-1) -> Tensor
 |      
 |      See :func:`torch.cross`
 |  
 |  ccuuddaa(...)
 |      cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor
 |      
 |      Returns a copy of this object in CUDA memory.
 |      
 |      If this object is already in CUDA memory and on the correct device,
 |      then no copy is performed and the original object is returned.
 |      
 |      Args:
 |          device (:class:`torch.device`): The destination GPU device.
 |              Defaults to the current CUDA device.
 |          non_blocking (bool): If ``True`` and the source is in pinned memory,
 |              the copy will be asynchronous with respect to the host.
 |              Otherwise, the argument has no effect. Default: ``False``.
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  ccuummmmaaxx(...)
 |      cummax(dim) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.cummax`
 |  
 |  ccuummmmiinn(...)
 |      cummin(dim) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.cummin`
 |  
 |  ccuummpprroodd(...)
 |      cumprod(dim, dtype=None) -> Tensor
 |      
 |      See :func:`torch.cumprod`
 |  
 |  ccuummpprroodd__(...)
 |      cumprod_(dim, dtype=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.cumprod`
 |  
 |  ccuummssuumm(...)
 |      cumsum(dim, dtype=None) -> Tensor
 |      
 |      See :func:`torch.cumsum`
 |  
 |  ccuummssuumm__(...)
 |      cumsum_(dim, dtype=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.cumsum`
 |  
 |  ddaattaa__ppttrr(...)
 |      data_ptr() -> int
 |      
 |      Returns the address of the first element of :attr:`self` tensor.
 |  
 |  ddeegg22rraadd(...)
 |      deg2rad() -> Tensor
 |      
 |      See :func:`torch.deg2rad`
 |  
 |  ddeegg22rraadd__(...)
 |      deg2rad_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.deg2rad`
 |  
 |  ddeennssee__ddiimm(...)
 |      dense_dim() -> int
 |      
 |      Return the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse tensor.
 |      
 |      See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
 |  
 |  ddeeqquuaannttiizzee(...)
 |      dequantize() -> Tensor
 |      
 |      Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
 |  
 |  ddeett(...)
 |      det() -> Tensor
 |      
 |      See :func:`torch.det`
 |  
 |  ddiiaagg(...)
 |      diag(diagonal=0) -> Tensor
 |      
 |      See :func:`torch.diag`
 |  
 |  ddiiaagg__eemmbbeedd(...)
 |      diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor
 |      
 |      See :func:`torch.diag_embed`
 |  
 |  ddiiaaggffllaatt(...)
 |      diagflat(offset=0) -> Tensor
 |      
 |      See :func:`torch.diagflat`
 |  
 |  ddiiaaggoonnaall(...)
 |      diagonal(offset=0, dim1=0, dim2=1) -> Tensor
 |      
 |      See :func:`torch.diagonal`
 |  
 |  ddiiffff(...)
 |      diff(n=1, dim=-1, prepend=None, append=None) -> Tensor
 |      
 |      See :func:`torch.diff`
 |  
 |  ddiiggaammmmaa(...)
 |      digamma() -> Tensor
 |      
 |      See :func:`torch.digamma`
 |  
 |  ddiiggaammmmaa__(...)
 |      digamma_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.digamma`
 |  
 |  ddiimm(...)
 |      dim() -> int
 |      
 |      Returns the number of dimensions of :attr:`self` tensor.
 |  
 |  ddiisstt(...)
 |      dist(other, p=2) -> Tensor
 |      
 |      See :func:`torch.dist`
 |  
 |  ddiivv(...)
 |      div(value, *, rounding_mode=None) -> Tensor
 |      
 |      See :func:`torch.div`
 |  
 |  ddiivv__(...)
 |      div_(value, *, rounding_mode=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.div`
 |  
 |  ddiivviiddee(...)
 |      divide(value, *, rounding_mode=None) -> Tensor
 |      
 |      See :func:`torch.divide`
 |  
 |  ddiivviiddee__(...)
 |      divide_(value, *, rounding_mode=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.divide`
 |  
 |  ddoott(...)
 |      dot(other) -> Tensor
 |      
 |      See :func:`torch.dot`
 |  
 |  ddoouubbllee(...)
 |      double(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  eeiigg(...)
 |      eig(eigenvectors=False) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.eig`
 |  
 |  eelleemmeenntt__ssiizzee(...)
 |      element_size() -> int
 |      
 |      Returns the size in bytes of an individual element.
 |      
 |      Example::
 |      
 |          >>> torch.tensor([]).element_size()
 |          4
 |          >>> torch.tensor([], dtype=torch.uint8).element_size()
 |          1
 |  
 |  eeqq(...)
 |      eq(other) -> Tensor
 |      
 |      See :func:`torch.eq`
 |  
 |  eeqq__(...)
 |      eq_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.eq`
 |  
 |  eeqquuaall(...)
 |      equal(other) -> bool
 |      
 |      See :func:`torch.equal`
 |  
 |  eerrff(...)
 |      erf() -> Tensor
 |      
 |      See :func:`torch.erf`
 |  
 |  eerrff__(...)
 |      erf_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.erf`
 |  
 |  eerrffcc(...)
 |      erfc() -> Tensor
 |      
 |      See :func:`torch.erfc`
 |  
 |  eerrffcc__(...)
 |      erfc_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.erfc`
 |  
 |  eerrffiinnvv(...)
 |      erfinv() -> Tensor
 |      
 |      See :func:`torch.erfinv`
 |  
 |  eerrffiinnvv__(...)
 |      erfinv_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.erfinv`
 |  
 |  eexxpp(...)
 |      exp() -> Tensor
 |      
 |      See :func:`torch.exp`
 |  
 |  eexxpp22(...)
 |      exp2() -> Tensor
 |      
 |      See :func:`torch.exp2`
 |  
 |  eexxpp22__(...)
 |      exp2_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.exp2`
 |  
 |  eexxpp__(...)
 |      exp_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.exp`
 |  
 |  eexxppaanndd(...)
 |      expand(*sizes) -> Tensor
 |      
 |      Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
 |      to a larger size.
 |      
 |      Passing -1 as the size for a dimension means not changing the size of
 |      that dimension.
 |      
 |      Tensor can be also expanded to a larger number of dimensions, and the
 |      new ones will be appended at the front. For the new dimensions, the
 |      size cannot be set to -1.
 |      
 |      Expanding a tensor does not allocate new memory, but only creates a
 |      new view on the existing tensor where a dimension of size one is
 |      expanded to a larger size by setting the ``stride`` to 0. Any dimension
 |      of size 1 can be expanded to an arbitrary value without allocating new
 |      memory.
 |      
 |      Args:
 |          *sizes (torch.Size or int...): the desired expanded size
 |      
 |      .. warning::
 |      
 |          More than one element of an expanded tensor may refer to a single
 |          memory location. As a result, in-place operations (especially ones that
 |          are vectorized) may result in incorrect behavior. If you need to write
 |          to the tensors, please clone them first.
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([[1], [2], [3]])
 |          >>> x.size()
 |          torch.Size([3, 1])
 |          >>> x.expand(3, 4)
 |          tensor([[ 1,  1,  1,  1],
 |                  [ 2,  2,  2,  2],
 |                  [ 3,  3,  3,  3]])
 |          >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
 |          tensor([[ 1,  1,  1,  1],
 |                  [ 2,  2,  2,  2],
 |                  [ 3,  3,  3,  3]])
 |  
 |  eexxppaanndd__aass(...)
 |      expand_as(other) -> Tensor
 |      
 |      Expand this tensor to the same size as :attr:`other`.
 |      ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.
 |      
 |      Please see :meth:`~Tensor.expand` for more information about ``expand``.
 |      
 |      Args:
 |          other (:class:`torch.Tensor`): The result tensor has the same size
 |              as :attr:`other`.
 |  
 |  eexxppmm11(...)
 |      expm1() -> Tensor
 |      
 |      See :func:`torch.expm1`
 |  
 |  eexxppmm11__(...)
 |      expm1_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.expm1`
 |  
 |  eexxppoonneennttiiaall__(...)
 |      exponential_(lambd=1, *, generator=None) -> Tensor
 |      
 |      Fills :attr:`self` tensor with elements drawn from the exponential distribution:
 |      
 |      .. math::
 |      
 |          f(x) = \lambda e^{-\lambda x}
 |  
 |  ffiillll__(...)
 |      fill_(value) -> Tensor
 |      
 |      Fills :attr:`self` tensor with the specified value.
 |  
 |  ffiillll__ddiiaaggoonnaall__(...)
 |      fill_diagonal_(fill_value, wrap=False) -> Tensor
 |      
 |      Fill the main diagonal of a tensor that has at least 2-dimensions.
 |      When dims>2, all dimensions of input must be of equal length.
 |      This function modifies the input tensor in-place, and returns the input tensor.
 |      
 |      Arguments:
 |          fill_value (Scalar): the fill value
 |          wrap (bool): the diagonal 'wrapped' after N columns for tall matrices.
 |      
 |      Example::
 |      
 |          >>> a = torch.zeros(3, 3)
 |          >>> a.fill_diagonal_(5)
 |          tensor([[5., 0., 0.],
 |                  [0., 5., 0.],
 |                  [0., 0., 5.]])
 |          >>> b = torch.zeros(7, 3)
 |          >>> b.fill_diagonal_(5)
 |          tensor([[5., 0., 0.],
 |                  [0., 5., 0.],
 |                  [0., 0., 5.],
 |                  [0., 0., 0.],
 |                  [0., 0., 0.],
 |                  [0., 0., 0.],
 |                  [0., 0., 0.]])
 |          >>> c = torch.zeros(7, 3)
 |          >>> c.fill_diagonal_(5, wrap=True)
 |          tensor([[5., 0., 0.],
 |                  [0., 5., 0.],
 |                  [0., 0., 5.],
 |                  [0., 0., 0.],
 |                  [5., 0., 0.],
 |                  [0., 5., 0.],
 |                  [0., 0., 5.]])
 |  
 |  ffiixx(...)
 |      fix() -> Tensor
 |      
 |      See :func:`torch.fix`.
 |  
 |  ffiixx__(...)
 |      fix_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.fix`
 |  
 |  ffllaatttteenn(...)
 |      flatten(input, start_dim=0, end_dim=-1) -> Tensor
 |      
 |      see :func:`torch.flatten`
 |  
 |  fflliipp(...)
 |      flip(dims) -> Tensor
 |      
 |      See :func:`torch.flip`
 |  
 |  fflliippllrr(...)
 |      fliplr() -> Tensor
 |      
 |      See :func:`torch.fliplr`
 |  
 |  fflliippuudd(...)
 |      flipud() -> Tensor
 |      
 |      See :func:`torch.flipud`
 |  
 |  ffllooaatt(...)
 |      float(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  ffllooaatt__ppoowweerr(...)
 |      float_power(exponent) -> Tensor
 |      
 |      See :func:`torch.float_power`
 |  
 |  ffllooaatt__ppoowweerr__(...)
 |      float_power_(exponent) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.float_power`
 |  
 |  fflloooorr(...)
 |      floor() -> Tensor
 |      
 |      See :func:`torch.floor`
 |  
 |  fflloooorr__(...)
 |      floor_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.floor`
 |  
 |  fflloooorr__ddiivviiddee(...)
 |      floor_divide(value) -> Tensor
 |      
 |      See :func:`torch.floor_divide`
 |  
 |  fflloooorr__ddiivviiddee__(...)
 |      floor_divide_(value) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.floor_divide`
 |  
 |  ffmmaaxx(...)
 |      fmax(other) -> Tensor
 |      
 |      See :func:`torch.fmax`
 |  
 |  ffmmiinn(...)
 |      fmin(other) -> Tensor
 |      
 |      See :func:`torch.fmin`
 |  
 |  ffmmoodd(...)
 |      fmod(divisor) -> Tensor
 |      
 |      See :func:`torch.fmod`
 |  
 |  ffmmoodd__(...)
 |      fmod_(divisor) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.fmod`
 |  
 |  ffrraacc(...)
 |      frac() -> Tensor
 |      
 |      See :func:`torch.frac`
 |  
 |  ffrraacc__(...)
 |      frac_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.frac`
 |  
 |  ggaatthheerr(...)
 |      gather(dim, index) -> Tensor
 |      
 |      See :func:`torch.gather`
 |  
 |  ggccdd(...)
 |      gcd(other) -> Tensor
 |      
 |      See :func:`torch.gcd`
 |  
 |  ggccdd__(...)
 |      gcd_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.gcd`
 |  
 |  ggee(...)
 |      ge(other) -> Tensor
 |      
 |      See :func:`torch.ge`.
 |  
 |  ggee__(...)
 |      ge_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.ge`.
 |  
 |  ggeeoommeettrriicc__(...)
 |      geometric_(p, *, generator=None) -> Tensor
 |      
 |      Fills :attr:`self` tensor with elements drawn from the geometric distribution:
 |      
 |      .. math::
 |      
 |          f(X=k) = p^{k - 1} (1 - p)
 |  
 |  ggeeqqrrff(...)
 |      geqrf() -> (Tensor, Tensor)
 |      
 |      See :func:`torch.geqrf`
 |  
 |  ggeerr(...)
 |      ger(vec2) -> Tensor
 |      
 |      See :func:`torch.ger`
 |  
 |  ggeett__ddeevviiccee(...)
 |      get_device() -> Device ordinal (Integer)
 |      
 |      For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
 |      For CPU tensors, an error is thrown.
 |      
 |      Example::
 |      
 |          >>> x = torch.randn(3, 4, 5, device='cuda:0')
 |          >>> x.get_device()
 |          0
 |          >>> x.cpu().get_device()  # RuntimeError: get_device is not implemented for type torch.FloatTensor
 |  
 |  ggrreeaatteerr(...)
 |      greater(other) -> Tensor
 |      
 |      See :func:`torch.greater`.
 |  
 |  ggrreeaatteerr__(...)
 |      greater_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.greater`.
 |  
 |  ggrreeaatteerr__eeqquuaall(...)
 |      greater_equal(other) -> Tensor
 |      
 |      See :func:`torch.greater_equal`.
 |  
 |  ggrreeaatteerr__eeqquuaall__(...)
 |      greater_equal_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.greater_equal`.
 |  
 |  ggtt(...)
 |      gt(other) -> Tensor
 |      
 |      See :func:`torch.gt`.
 |  
 |  ggtt__(...)
 |      gt_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.gt`.
 |  
 |  hhaallff(...)
 |      half(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  hhaarrddsshhrriinnkk(...)
 |      hardshrink(lambd=0.5) -> Tensor
 |      
 |      See :func:`torch.nn.functional.hardshrink`
 |  
 |  hhaass__nnaammeess(...)
 |      Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``.
 |  
 |  hheeaavviissiiddee(...)
 |      heaviside(values) -> Tensor
 |      
 |      See :func:`torch.heaviside`
 |  
 |  hheeaavviissiiddee__(...)
 |      heaviside_(values) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.heaviside`
 |  
 |  hhiissttcc(...)
 |      histc(bins=100, min=0, max=0) -> Tensor
 |      
 |      See :func:`torch.histc`
 |  
 |  hhyyppoott(...)
 |      hypot(other) -> Tensor
 |      
 |      See :func:`torch.hypot`
 |  
 |  hhyyppoott__(...)
 |      hypot_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.hypot`
 |  
 |  ii00(...)
 |      i0() -> Tensor
 |      
 |      See :func:`torch.i0`
 |  
 |  ii00__(...)
 |      i0_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.i0`
 |  
 |  iiggaammmmaa(...)
 |      igamma(other) -> Tensor
 |      
 |      See :func:`torch.igamma`
 |  
 |  iiggaammmmaa__(...)
 |      igamma_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.igamma`
 |  
 |  iiggaammmmaacc(...)
 |      igammac(other) -> Tensor
 |      See :func:`torch.igammac`
 |  
 |  iiggaammmmaacc__(...)
 |      igammac_(other) -> Tensor
 |      In-place version of :meth:`~Tensor.igammac`
 |  
 |  iinnddeexx__aadddd(...)
 |      index_add(tensor1, dim, index, tensor2) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.index_add_`.
 |      `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_add_`.
 |  
 |  iinnddeexx__aadddd__(...)
 |      index_add_(dim, index, tensor) -> Tensor
 |      
 |      Accumulate the elements of :attr:`tensor` into the :attr:`self` tensor by adding
 |      to the indices in the order given in :attr:`index`. For example, if ``dim == 0``
 |      and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is added to the
 |      ``j``\ th row of :attr:`self`.
 |      
 |      The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
 |      length of :attr:`index` (which must be a vector), and all other dimensions must
 |      match :attr:`self`, or an error will be raised.
 |      
 |      Note:
 |          This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.
 |      
 |      Args:
 |          dim (int): dimension along which to index
 |          index (IntTensor or LongTensor): indices of :attr:`tensor` to select from
 |          tensor (Tensor): the tensor containing values to add
 |      
 |      Example::
 |      
 |          >>> x = torch.ones(5, 3)
 |          >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
 |          >>> index = torch.tensor([0, 4, 2])
 |          >>> x.index_add_(0, index, t)
 |          tensor([[  2.,   3.,   4.],
 |                  [  1.,   1.,   1.],
 |                  [  8.,   9.,  10.],
 |                  [  1.,   1.,   1.],
 |                  [  5.,   6.,   7.]])
 |  
 |  iinnddeexx__ccooppyy(...)
 |      index_copy(tensor1, dim, index, tensor2) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.index_copy_`.
 |      `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_copy_`.
 |  
 |  iinnddeexx__ccooppyy__(...)
 |      index_copy_(dim, index, tensor) -> Tensor
 |      
 |      Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
 |      the indices in the order given in :attr:`index`. For example, if ``dim == 0``
 |      and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
 |      ``j``\ th row of :attr:`self`.
 |      
 |      The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
 |      length of :attr:`index` (which must be a vector), and all other dimensions must
 |      match :attr:`self`, or an error will be raised.
 |      
 |      .. note::
 |          If :attr:`index` contains duplicate entries, multiple elements from
 |          :attr:`tensor` will be copied to the same index of :attr:`self`. The result
 |          is nondeterministic since it depends on which copy occurs last.
 |      
 |      Args:
 |          dim (int): dimension along which to index
 |          index (LongTensor): indices of :attr:`tensor` to select from
 |          tensor (Tensor): the tensor containing values to copy
 |      
 |      Example::
 |      
 |          >>> x = torch.zeros(5, 3)
 |          >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
 |          >>> index = torch.tensor([0, 4, 2])
 |          >>> x.index_copy_(0, index, t)
 |          tensor([[ 1.,  2.,  3.],
 |                  [ 0.,  0.,  0.],
 |                  [ 7.,  8.,  9.],
 |                  [ 0.,  0.,  0.],
 |                  [ 4.,  5.,  6.]])
 |  
 |  iinnddeexx__ffiillll(...)
 |      index_fill(tensor1, dim, index, value) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.index_fill_`.
 |      `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_fill_`.
 |  
 |  iinnddeexx__ffiillll__(...)
 |      index_fill_(dim, index, val) -> Tensor
 |      
 |      Fills the elements of the :attr:`self` tensor with value :attr:`val` by
 |      selecting the indices in the order given in :attr:`index`.
 |      
 |      Args:
 |          dim (int): dimension along which to index
 |          index (LongTensor): indices of :attr:`self` tensor to fill in
 |          val (float): the value to fill with
 |      
 |      Example::
 |          >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
 |          >>> index = torch.tensor([0, 2])
 |          >>> x.index_fill_(1, index, -1)
 |          tensor([[-1.,  2., -1.],
 |                  [-1.,  5., -1.],
 |                  [-1.,  8., -1.]])
 |  
 |  iinnddeexx__ppuutt(...)
 |      index_put(tensor1, indices, values, accumulate=False) -> Tensor
 |      
 |      Out-place version of :meth:`~Tensor.index_put_`.
 |      `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_put_`.
 |  
 |  iinnddeexx__ppuutt__(...)
 |      index_put_(indices, values, accumulate=False) -> Tensor
 |      
 |      Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
 |      the indices specified in :attr:`indices` (which is a tuple of Tensors). The
 |      expression ``tensor.index_put_(indices, values)`` is equivalent to
 |      ``tensor[indices] = values``. Returns :attr:`self`.
 |      
 |      If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
 |      :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
 |      contain duplicate elements.
 |      
 |      Args:
 |          indices (tuple of LongTensor): tensors used to index into `self`.
 |          values (Tensor): tensor of same dtype as `self`.
 |          accumulate (bool): whether to accumulate into self
 |  
 |  iinnddeexx__sseelleecctt(...)
 |      index_select(dim, index) -> Tensor
 |      
 |      See :func:`torch.index_select`
 |  
 |  iinnddiicceess(...)
 |      indices() -> Tensor
 |      
 |      Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse COO tensor.
 |      
 |      See also :meth:`Tensor.values`.
 |      
 |      .. note::
 |        This method can only be called on a coalesced sparse tensor. See
 |        :meth:`Tensor.coalesce` for details.
 |  
 |  iinnnneerr(...)
 |      inner(other) -> Tensor
 |      
 |      See :func:`torch.inner`.
 |  
 |  iinntt(...)
 |      int(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  iinntt__rreepprr(...)
 |      int_repr() -> Tensor
 |      
 |      Given a quantized Tensor,
 |      ``self.int_repr()`` returns a CPU Tensor with uint8_t as data type that stores the
 |      underlying uint8_t values of the given Tensor.
 |  
 |  iinnvveerrssee(...)
 |      inverse() -> Tensor
 |      
 |      See :func:`torch.inverse`
 |  
 |  iiss__ccooaalleesscceedd(...)
 |      is_coalesced() -> bool
 |      
 |      Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor
 |      <sparse-coo-docs>` that is coalesced, ``False`` otherwise.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse COO tensor.
 |      
 |      See :meth:`coalesce` and :ref:`uncoalesced tensors <sparse-uncoalesced-coo-docs>`.
 |  
 |  iiss__ccoommpplleexx(...)
 |      is_complex() -> bool
 |      
 |      Returns True if the data type of :attr:`self` is a complex data type.
 |  
 |  iiss__ccoonnttiigguuoouuss(...)
 |      is_contiguous(memory_format=torch.contiguous_format) -> bool
 |      
 |      Returns True if :attr:`self` tensor is contiguous in memory in the order specified
 |      by memory format.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): Specifies memory allocation
 |              order. Default: ``torch.contiguous_format``.
 |  
 |  iiss__ddiissttrriibbuutteedd(...)
 |  
 |  iiss__ffllooaattiinngg__ppooiinntt(...)
 |      is_floating_point() -> bool
 |      
 |      Returns True if the data type of :attr:`self` is a floating point data type.
 |  
 |  iiss__nnoonnzzeerroo(...)
 |  
 |  iiss__ppiinnnneedd(...)
 |      Returns true if this tensor resides in pinned memory.
 |  
 |  iiss__ssaammee__ssiizzee(...)
 |  
 |  iiss__sseett__ttoo(...)
 |      is_set_to(tensor) -> bool
 |      
 |      Returns True if both tensors are pointing to the exact same memory (same
 |      storage, offset, size and stride).
 |  
 |  iiss__ssiiggnneedd(...)
 |      is_signed() -> bool
 |      
 |      Returns True if the data type of :attr:`self` is a signed data type.
 |  
 |  iisscclloossee(...)
 |      isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
 |      
 |      See :func:`torch.isclose`
 |  
 |  iissffiinniittee(...)
 |      isfinite() -> Tensor
 |      
 |      See :func:`torch.isfinite`
 |  
 |  iissiinnff(...)
 |      isinf() -> Tensor
 |      
 |      See :func:`torch.isinf`
 |  
 |  iissnnaann(...)
 |      isnan() -> Tensor
 |      
 |      See :func:`torch.isnan`
 |  
 |  iissnneeggiinnff(...)
 |      isneginf() -> Tensor
 |      
 |      See :func:`torch.isneginf`
 |  
 |  iissppoossiinnff(...)
 |      isposinf() -> Tensor
 |      
 |      See :func:`torch.isposinf`
 |  
 |  iissrreeaall(...)
 |      isreal() -> Tensor
 |      
 |      See :func:`torch.isreal`
 |  
 |  iitteemm(...)
 |      item() -> number
 |      
 |      Returns the value of this tensor as a standard Python number. This only works
 |      for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.
 |      
 |      This operation is not differentiable.
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([1.0])
 |          >>> x.item()
 |          1.0
 |  
 |  kkrroonn(...)
 |      kron(other) -> Tensor
 |      
 |      See :func:`torch.kron`
 |  
 |  kktthhvvaalluuee(...)
 |      kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.kthvalue`
 |  
 |  llccmm(...)
 |      lcm(other) -> Tensor
 |      
 |      See :func:`torch.lcm`
 |  
 |  llccmm__(...)
 |      lcm_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.lcm`
 |  
 |  llddeexxpp(...)
 |      ldexp(other) -> Tensor
 |      
 |      See :func:`torch.ldexp`
 |  
 |  llddeexxpp__(...)
 |      ldexp_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.ldexp`
 |  
 |  llee(...)
 |      le(other) -> Tensor
 |      
 |      See :func:`torch.le`.
 |  
 |  llee__(...)
 |      le_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.le`.
 |  
 |  lleerrpp(...)
 |      lerp(end, weight) -> Tensor
 |      
 |      See :func:`torch.lerp`
 |  
 |  lleerrpp__(...)
 |      lerp_(end, weight) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.lerp`
 |  
 |  lleessss(...)
 |      lt(other) -> Tensor
 |      
 |      See :func:`torch.less`.
 |  
 |  lleessss__(...)
 |      less_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.less`.
 |  
 |  lleessss__eeqquuaall(...)
 |      less_equal(other) -> Tensor
 |      
 |      See :func:`torch.less_equal`.
 |  
 |  lleessss__eeqquuaall__(...)
 |      less_equal_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.less_equal`.
 |  
 |  llggaammmmaa(...)
 |      lgamma() -> Tensor
 |      
 |      See :func:`torch.lgamma`
 |  
 |  llggaammmmaa__(...)
 |      lgamma_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.lgamma`
 |  
 |  lloogg(...)
 |      log() -> Tensor
 |      
 |      See :func:`torch.log`
 |  
 |  lloogg1100(...)
 |      log10() -> Tensor
 |      
 |      See :func:`torch.log10`
 |  
 |  lloogg1100__(...)
 |      log10_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.log10`
 |  
 |  lloogg11pp(...)
 |      log1p() -> Tensor
 |      
 |      See :func:`torch.log1p`
 |  
 |  lloogg11pp__(...)
 |      log1p_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.log1p`
 |  
 |  lloogg22(...)
 |      log2() -> Tensor
 |      
 |      See :func:`torch.log2`
 |  
 |  lloogg22__(...)
 |      log2_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.log2`
 |  
 |  lloogg__(...)
 |      log_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.log`
 |  
 |  lloogg__nnoorrmmaall__(...)
 |      log_normal_(mean=1, std=2, *, generator=None)
 |      
 |      Fills :attr:`self` tensor with numbers samples from the log-normal distribution
 |      parameterized by the given mean :math:`\mu` and standard deviation
 |      :math:`\sigma`. Note that :attr:`mean` and :attr:`std` are the mean and
 |      standard deviation of the underlying normal distribution, and not of the
 |      returned distribution:
 |      
 |      .. math::
 |      
 |          f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\ e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
 |  
 |  lloogg__ssooffttmmaaxx(...)
 |  
 |  llooggaaddddeexxpp(...)
 |      logaddexp(other) -> Tensor
 |      
 |      See :func:`torch.logaddexp`
 |  
 |  llooggaaddddeexxpp22(...)
 |      logaddexp2(other) -> Tensor
 |      
 |      See :func:`torch.logaddexp2`
 |  
 |  llooggccuummssuummeexxpp(...)
 |      logcumsumexp(dim) -> Tensor
 |      
 |      See :func:`torch.logcumsumexp`
 |  
 |  llooggddeett(...)
 |      logdet() -> Tensor
 |      
 |      See :func:`torch.logdet`
 |  
 |  llooggiiccaall__aanndd(...)
 |      logical_and() -> Tensor
 |      
 |      See :func:`torch.logical_and`
 |  
 |  llooggiiccaall__aanndd__(...)
 |      logical_and_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.logical_and`
 |  
 |  llooggiiccaall__nnoott(...)
 |      logical_not() -> Tensor
 |      
 |      See :func:`torch.logical_not`
 |  
 |  llooggiiccaall__nnoott__(...)
 |      logical_not_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.logical_not`
 |  
 |  llooggiiccaall__oorr(...)
 |      logical_or() -> Tensor
 |      
 |      See :func:`torch.logical_or`
 |  
 |  llooggiiccaall__oorr__(...)
 |      logical_or_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.logical_or`
 |  
 |  llooggiiccaall__xxoorr(...)
 |      logical_xor() -> Tensor
 |      
 |      See :func:`torch.logical_xor`
 |  
 |  llooggiiccaall__xxoorr__(...)
 |      logical_xor_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.logical_xor`
 |  
 |  llooggiitt(...)
 |      logit() -> Tensor
 |      
 |      See :func:`torch.logit`
 |  
 |  llooggiitt__(...)
 |      logit_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.logit`
 |  
 |  llooggssuummeexxpp(...)
 |      logsumexp(dim, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.logsumexp`
 |  
 |  lloonngg(...)
 |      long(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  llssttssqq(...)
 |      lstsq(A) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.lstsq`
 |  
 |  lltt(...)
 |      lt(other) -> Tensor
 |      
 |      See :func:`torch.lt`.
 |  
 |  lltt__(...)
 |      lt_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.lt`.
 |  
 |  lluu__ssoollvvee(...)
 |      lu_solve(LU_data, LU_pivots) -> Tensor
 |      
 |      See :func:`torch.lu_solve`
 |  
 |  mmaapp22__(...)
 |  
 |  mmaapp__(...)
 |      map_(tensor, callable)
 |      
 |      Applies :attr:`callable` for each element in :attr:`self` tensor and the given
 |      :attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
 |      the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.
 |      
 |      The :attr:`callable` should have the signature::
 |      
 |          def callable(a, b) -> number
 |  
 |  mmaasskkeedd__ffiillll(...)
 |      masked_fill(mask, value) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.masked_fill_`
 |  
 |  mmaasskkeedd__ffiillll__(...)
 |      masked_fill_(mask, value)
 |      
 |      Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
 |      True. The shape of :attr:`mask` must be
 |      :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
 |      tensor.
 |      
 |      Args:
 |          mask (BoolTensor): the boolean mask
 |          value (float): the value to fill in with
 |  
 |  mmaasskkeedd__ssccaatttteerr(...)
 |      masked_scatter(mask, tensor) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.masked_scatter_`
 |  
 |  mmaasskkeedd__ssccaatttteerr__(...)
 |      masked_scatter_(mask, source)
 |      
 |      Copies elements from :attr:`source` into :attr:`self` tensor at positions where
 |      the :attr:`mask` is True.
 |      The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
 |      with the shape of the underlying tensor. The :attr:`source` should have at least
 |      as many elements as the number of ones in :attr:`mask`
 |      
 |      Args:
 |          mask (BoolTensor): the boolean mask
 |          source (Tensor): the tensor to copy from
 |      
 |      .. note::
 |      
 |          The :attr:`mask` operates on the :attr:`self` tensor, not on the given
 |          :attr:`source` tensor.
 |  
 |  mmaasskkeedd__sseelleecctt(...)
 |      masked_select(mask) -> Tensor
 |      
 |      See :func:`torch.masked_select`
 |  
 |  mmaattmmuull(...)
 |      matmul(tensor2) -> Tensor
 |      
 |      See :func:`torch.matmul`
 |  
 |  mmaattrriixx__eexxpp(...)
 |      matrix_exp() -> Tensor
 |      
 |      See :func:`torch.matrix_exp`
 |  
 |  mmaattrriixx__ppoowweerr(...)
 |      matrix_power(n) -> Tensor
 |      
 |      See :func:`torch.matrix_power`
 |  
 |  mmaaxx(...)
 |      max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)
 |      
 |      See :func:`torch.max`
 |  
 |  mmaaxxiimmuumm(...)
 |      maximum(other) -> Tensor
 |      
 |      See :func:`torch.maximum`
 |  
 |  mmeeaann(...)
 |      mean(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)
 |      
 |      See :func:`torch.mean`
 |  
 |  mmeeddiiaann(...)
 |      median(dim=None, keepdim=False) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.median`
 |  
 |  mmiinn(...)
 |      min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)
 |      
 |      See :func:`torch.min`
 |  
 |  mmiinniimmuumm(...)
 |      minimum(other) -> Tensor
 |      
 |      See :func:`torch.minimum`
 |  
 |  mmmm(...)
 |      mm(mat2) -> Tensor
 |      
 |      See :func:`torch.mm`
 |  
 |  mmooddee(...)
 |      mode(dim=None, keepdim=False) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.mode`
 |  
 |  mmoovveeaaxxiiss(...)
 |      moveaxis(source, destination) -> Tensor
 |      
 |      See :func:`torch.moveaxis`
 |  
 |  mmoovveeddiimm(...)
 |      movedim(source, destination) -> Tensor
 |      
 |      See :func:`torch.movedim`
 |  
 |  mmssoorrtt(...)
 |      msort() -> Tensor
 |      
 |      See :func:`torch.msort`
 |  
 |  mmuull(...)
 |      mul(value) -> Tensor
 |      
 |      See :func:`torch.mul`.
 |  
 |  mmuull__(...)
 |      mul_(value) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.mul`.
 |  
 |  mmuullttiinnoommiiaall(...)
 |      multinomial(num_samples, replacement=False, *, generator=None) -> Tensor
 |      
 |      See :func:`torch.multinomial`
 |  
 |  mmuullttiippllyy(...)
 |      multiply(value) -> Tensor
 |      
 |      See :func:`torch.multiply`.
 |  
 |  mmuullttiippllyy__(...)
 |      multiply_(value) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.multiply`.
 |  
 |  mmvv(...)
 |      mv(vec) -> Tensor
 |      
 |      See :func:`torch.mv`
 |  
 |  mmvvllggaammmmaa(...)
 |      mvlgamma(p) -> Tensor
 |      
 |      See :func:`torch.mvlgamma`
 |  
 |  mmvvllggaammmmaa__(...)
 |      mvlgamma_(p) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.mvlgamma`
 |  
 |  nnaann__ttoo__nnuumm(...)
 |      nan_to_num(nan=0.0, posinf=None, neginf=None) -> Tensor
 |      
 |      See :func:`torch.nan_to_num`.
 |  
 |  nnaann__ttoo__nnuumm__(...)
 |      nan_to_num_(nan=0.0, posinf=None, neginf=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.nan_to_num`.
 |  
 |  nnaannmmeeddiiaann(...)
 |      nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.nanmedian`
 |  
 |  nnaannqquuaannttiillee(...)
 |      nanquantile(q, dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.nanquantile`
 |  
 |  nnaannssuumm(...)
 |      nansum(dim=None, keepdim=False, dtype=None) -> Tensor
 |      
 |      See :func:`torch.nansum`
 |  
 |  nnaarrrrooww(...)
 |      narrow(dimension, start, length) -> Tensor
 |      
 |      See :func:`torch.narrow`
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 |          >>> x.narrow(0, 0, 2)
 |          tensor([[ 1,  2,  3],
 |                  [ 4,  5,  6]])
 |          >>> x.narrow(1, 1, 2)
 |          tensor([[ 2,  3],
 |                  [ 5,  6],
 |                  [ 8,  9]])
 |  
 |  nnaarrrrooww__ccooppyy(...)
 |      narrow_copy(dimension, start, length) -> Tensor
 |      
 |      Same as :meth:`Tensor.narrow` except returning a copy rather
 |      than shared storage.  This is primarily for sparse tensors, which
 |      do not have a shared-storage narrow method.  Calling ```narrow_copy``
 |      with ```dimemsion > self.sparse_dim()``` will return a copy with the
 |      relevant dense dimension narrowed, and ```self.shape``` updated accordingly.
 |  
 |  nnddiimmeennssiioonn(...)
 |      ndimension() -> int
 |      
 |      Alias for :meth:`~Tensor.dim()`
 |  
 |  nnee(...)
 |      ne(other) -> Tensor
 |      
 |      See :func:`torch.ne`.
 |  
 |  nnee__(...)
 |      ne_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.ne`.
 |  
 |  nneegg(...)
 |      neg() -> Tensor
 |      
 |      See :func:`torch.neg`
 |  
 |  nneegg__(...)
 |      neg_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.neg`
 |  
 |  nneeggaattiivvee(...)
 |      negative() -> Tensor
 |      
 |      See :func:`torch.negative`
 |  
 |  nneeggaattiivvee__(...)
 |      negative_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.negative`
 |  
 |  nneelleemmeenntt(...)
 |      nelement() -> int
 |      
 |      Alias for :meth:`~Tensor.numel`
 |  
 |  nneeww(...)
 |  
 |  nneeww__eemmppttyy(...)
 |      new_empty(size, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a Tensor of size :attr:`size` filled with uninitialized data.
 |      By default, the returned Tensor has the same :class:`torch.dtype` and
 |      :class:`torch.device` as this tensor.
 |      
 |      Args:
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.ones(())
 |          >>> tensor.new_empty((2, 3))
 |          tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
 |                  [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
 |  
 |  nneeww__eemmppttyy__ssttrriiddeedd(...)
 |      new_empty_strided(size, stride, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
 |      uninitialized data. By default, the returned Tensor has the same
 |      :class:`torch.dtype` and :class:`torch.device` as this tensor.
 |      
 |      Args:
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.ones(())
 |          >>> tensor.new_empty_strided((2, 3), (3, 1))
 |          tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
 |                  [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
 |  
 |  nneeww__ffuullll(...)
 |      new_full(size, fill_value, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
 |      By default, the returned Tensor has the same :class:`torch.dtype` and
 |      :class:`torch.device` as this tensor.
 |      
 |      Args:
 |          fill_value (scalar): the number to fill the output tensor with.
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.ones((2,), dtype=torch.float64)
 |          >>> tensor.new_full((3, 4), 3.141592)
 |          tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
 |                  [ 3.1416,  3.1416,  3.1416,  3.1416],
 |                  [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
 |  
 |  nneeww__oonneess(...)
 |      new_ones(size, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a Tensor of size :attr:`size` filled with ``1``.
 |      By default, the returned Tensor has the same :class:`torch.dtype` and
 |      :class:`torch.device` as this tensor.
 |      
 |      Args:
 |          size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
 |              shape of the output tensor.
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.tensor((), dtype=torch.int32)
 |          >>> tensor.new_ones((2, 3))
 |          tensor([[ 1,  1,  1],
 |                  [ 1,  1,  1]], dtype=torch.int32)
 |  
 |  nneeww__tteennssoorr(...)
 |      new_tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a new Tensor with :attr:`data` as the tensor data.
 |      By default, the returned Tensor has the same :class:`torch.dtype` and
 |      :class:`torch.device` as this tensor.
 |      
 |      .. warning::
 |      
 |          :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
 |          ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
 |          or :func:`torch.Tensor.detach`.
 |          If you have a numpy array and want to avoid a copy, use
 |          :func:`torch.from_numpy`.
 |      
 |      .. warning::
 |      
 |          When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
 |          and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.clone().detach()``
 |          and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
 |          The equivalents using ``clone()`` and ``detach()`` are recommended.
 |      
 |      Args:
 |          data (array_like): The returned Tensor copies :attr:`data`.
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.ones((2,), dtype=torch.int8)
 |          >>> data = [[0, 1], [2, 3]]
 |          >>> tensor.new_tensor(data)
 |          tensor([[ 0,  1],
 |                  [ 2,  3]], dtype=torch.int8)
 |  
 |  nneeww__zzeerrooss(...)
 |      new_zeros(size, dtype=None, device=None, requires_grad=False) -> Tensor
 |      
 |      Returns a Tensor of size :attr:`size` filled with ``0``.
 |      By default, the returned Tensor has the same :class:`torch.dtype` and
 |      :class:`torch.device` as this tensor.
 |      
 |      Args:
 |          size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
 |              shape of the output tensor.
 |          dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
 |              Default: if None, same :class:`torch.dtype` as this tensor.
 |          device (:class:`torch.device`, optional): the desired device of returned tensor.
 |              Default: if None, same :class:`torch.device` as this tensor.
 |          requires_grad (bool, optional): If autograd should record operations on the
 |              returned tensor. Default: ``False``.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.tensor((), dtype=torch.float64)
 |          >>> tensor.new_zeros((2, 3))
 |          tensor([[ 0.,  0.,  0.],
 |                  [ 0.,  0.,  0.]], dtype=torch.float64)
 |  
 |  nneexxttaafftteerr(...)
 |      nextafter(other) -> Tensor
 |      See :func:`torch.nextafter`
 |  
 |  nneexxttaafftteerr__(...)
 |      nextafter_(other) -> Tensor
 |      In-place version of :meth:`~Tensor.nextafter`
 |  
 |  nnoonnzzeerroo(...)
 |      nonzero() -> LongTensor
 |      
 |      See :func:`torch.nonzero`
 |  
 |  nnoorrmmaall__(...)
 |      normal_(mean=0, std=1, *, generator=None) -> Tensor
 |      
 |      Fills :attr:`self` tensor with elements samples from the normal distribution
 |      parameterized by :attr:`mean` and :attr:`std`.
 |  
 |  nnoott__eeqquuaall(...)
 |      not_equal(other) -> Tensor
 |      
 |      See :func:`torch.not_equal`.
 |  
 |  nnoott__eeqquuaall__(...)
 |      not_equal_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.not_equal`.
 |  
 |  nnuummeell(...)
 |      numel() -> int
 |      
 |      See :func:`torch.numel`
 |  
 |  nnuummppyy(...)
 |      numpy() -> numpy.ndarray
 |      
 |      Returns :attr:`self` tensor as a NumPy :class:`ndarray`. This tensor and the
 |      returned :class:`ndarray` share the same underlying storage. Changes to
 |      :attr:`self` tensor will be reflected in the :class:`ndarray` and vice versa.
 |  
 |  oorrggqqrr(...)
 |      orgqr(input2) -> Tensor
 |      
 |      See :func:`torch.orgqr`
 |  
 |  oorrmmqqrr(...)
 |      ormqr(input2, input3, left=True, transpose=False) -> Tensor
 |      
 |      See :func:`torch.ormqr`
 |  
 |  oouutteerr(...)
 |      outer(vec2) -> Tensor
 |      
 |      See :func:`torch.outer`.
 |  
 |  ppeerrmmuuttee(...)
 |      permute(*dims) -> Tensor
 |      
 |      Returns a view of the original tensor with its dimensions permuted.
 |      
 |      Args:
 |          *dims (int...): The desired ordering of dimensions
 |      
 |      Example:
 |          >>> x = torch.randn(2, 3, 5)
 |          >>> x.size()
 |          torch.Size([2, 3, 5])
 |          >>> x.permute(2, 0, 1).size()
 |          torch.Size([5, 2, 3])
 |  
 |  ppiinn__mmeemmoorryy(...)
 |      pin_memory() -> Tensor
 |      
 |      Copies the tensor to pinned memory, if it's not already pinned.
 |  
 |  ppiinnvveerrssee(...)
 |      pinverse() -> Tensor
 |      
 |      See :func:`torch.pinverse`
 |  
 |  ppoollyyggaammmmaa(...)
 |      polygamma(n) -> Tensor
 |      
 |      See :func:`torch.polygamma`
 |  
 |  ppoollyyggaammmmaa__(...)
 |      polygamma_(n) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.polygamma`
 |  
 |  ppooww(...)
 |      pow(exponent) -> Tensor
 |      
 |      See :func:`torch.pow`
 |  
 |  ppooww__(...)
 |      pow_(exponent) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.pow`
 |  
 |  pprreelluu(...)
 |  
 |  pprroodd(...)
 |      prod(dim=None, keepdim=False, dtype=None) -> Tensor
 |      
 |      See :func:`torch.prod`
 |  
 |  ppuutt__(...)
 |      put_(indices, tensor, accumulate=False) -> Tensor
 |      
 |      Copies the elements from :attr:`tensor` into the positions specified by
 |      indices. For the purpose of indexing, the :attr:`self` tensor is treated as if
 |      it were a 1-D tensor.
 |      
 |      If :attr:`accumulate` is ``True``, the elements in :attr:`tensor` are added to
 |      :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
 |      contain duplicate elements.
 |      
 |      Args:
 |          indices (LongTensor): the indices into self
 |          tensor (Tensor): the tensor containing values to copy from
 |          accumulate (bool): whether to accumulate into self
 |      
 |      Example::
 |      
 |          >>> src = torch.tensor([[4, 3, 5],
 |          ...                     [6, 7, 8]])
 |          >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
 |          tensor([[  4,   9,   5],
 |                  [ 10,   7,   8]])
 |  
 |  qq__ppeerr__cchhaannnneell__aaxxiiss(...)
 |      q_per_channel_axis() -> int
 |      
 |      Given a Tensor quantized by linear (affine) per-channel quantization,
 |      returns the index of dimension on which per-channel quantization is applied.
 |  
 |  qq__ppeerr__cchhaannnneell__ssccaalleess(...)
 |      q_per_channel_scales() -> Tensor
 |      
 |      Given a Tensor quantized by linear (affine) per-channel quantization,
 |      returns a Tensor of scales of the underlying quantizer. It has the number of
 |      elements that matches the corresponding dimensions (from q_per_channel_axis) of
 |      the tensor.
 |  
 |  qq__ppeerr__cchhaannnneell__zzeerroo__ppooiinnttss(...)
 |      q_per_channel_zero_points() -> Tensor
 |      
 |      Given a Tensor quantized by linear (affine) per-channel quantization,
 |      returns a tensor of zero_points of the underlying quantizer. It has the number of
 |      elements that matches the corresponding dimensions (from q_per_channel_axis) of
 |      the tensor.
 |  
 |  qq__ssccaallee(...)
 |      q_scale() -> float
 |      
 |      Given a Tensor quantized by linear(affine) quantization,
 |      returns the scale of the underlying quantizer().
 |  
 |  qq__zzeerroo__ppooiinntt(...)
 |      q_zero_point() -> int
 |      
 |      Given a Tensor quantized by linear(affine) quantization,
 |      returns the zero_point of the underlying quantizer().
 |  
 |  qqrr(...)
 |      qr(some=True) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.qr`
 |  
 |  qqsscchheemmee(...)
 |      qscheme() -> torch.qscheme
 |      
 |      Returns the quantization scheme of a given QTensor.
 |  
 |  qquuaannttiillee(...)
 |      quantile(q, dim=None, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.quantile`
 |  
 |  rraadd22ddeegg(...)
 |      rad2deg() -> Tensor
 |      
 |      See :func:`torch.rad2deg`
 |  
 |  rraadd22ddeegg__(...)
 |      rad2deg_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.rad2deg`
 |  
 |  rraannddoomm__(...)
 |      random_(from=0, to=None, *, generator=None) -> Tensor
 |      
 |      Fills :attr:`self` tensor with numbers sampled from the discrete uniform
 |      distribution over ``[from, to - 1]``. If not specified, the values are usually
 |      only bounded by :attr:`self` tensor's data type. However, for floating point
 |      types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
 |      value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
 |      will be uniform in ``[0, 2^53]``.
 |  
 |  rraavveell(...)
 |      ravel(input) -> Tensor
 |      
 |      see :func:`torch.ravel`
 |  
 |  rreecciipprrooccaall(...)
 |      reciprocal() -> Tensor
 |      
 |      See :func:`torch.reciprocal`
 |  
 |  rreecciipprrooccaall__(...)
 |      reciprocal_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.reciprocal`
 |  
 |  rreeccoorrdd__ssttrreeaamm(...)
 |      record_stream(stream)
 |      
 |      Ensures that the tensor memory is not reused for another tensor until all
 |      current work queued on :attr:`stream` are complete.
 |      
 |      .. note::
 |      
 |          The caching allocator is aware of only the stream where a tensor was
 |          allocated. Due to the awareness, it already correctly manages the life
 |          cycle of tensors on only one stream. But if a tensor is used on a stream
 |          different from the stream of origin, the allocator might reuse the memory
 |          unexpectedly. Calling this method lets the allocator know which streams
 |          have used the tensor.
 |  
 |  rreelluu(...)
 |  
 |  rreelluu__(...)
 |  
 |  rreemmaaiinnddeerr(...)
 |      remainder(divisor) -> Tensor
 |      
 |      See :func:`torch.remainder`
 |  
 |  rreemmaaiinnddeerr__(...)
 |      remainder_(divisor) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.remainder`
 |  
 |  rreennoorrmm(...)
 |      renorm(p, dim, maxnorm) -> Tensor
 |      
 |      See :func:`torch.renorm`
 |  
 |  rreennoorrmm__(...)
 |      renorm_(p, dim, maxnorm) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.renorm`
 |  
 |  rreeppeeaatt(...)
 |      repeat(*sizes) -> Tensor
 |      
 |      Repeats this tensor along the specified dimensions.
 |      
 |      Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.
 |      
 |      .. warning::
 |      
 |          :meth:`~Tensor.repeat` behaves differently from
 |          `numpy.repeat <https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html>`_,
 |          but is more similar to
 |          `numpy.tile <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_.
 |          For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.
 |      
 |      Args:
 |          sizes (torch.Size or int...): The number of times to repeat this tensor along each
 |              dimension
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([1, 2, 3])
 |          >>> x.repeat(4, 2)
 |          tensor([[ 1,  2,  3,  1,  2,  3],
 |                  [ 1,  2,  3,  1,  2,  3],
 |                  [ 1,  2,  3,  1,  2,  3],
 |                  [ 1,  2,  3,  1,  2,  3]])
 |          >>> x.repeat(4, 2, 1).size()
 |          torch.Size([4, 2, 3])
 |  
 |  rreeppeeaatt__iinntteerrlleeaavvee(...)
 |      repeat_interleave(repeats, dim=None) -> Tensor
 |      
 |      See :func:`torch.repeat_interleave`.
 |  
 |  rreeqquuiirreess__ggrraadd__(...)
 |      requires_grad_(requires_grad=True) -> Tensor
 |      
 |      Change if autograd should record operations on this tensor: sets this tensor's
 |      :attr:`requires_grad` attribute in-place. Returns this tensor.
 |      
 |      :func:`requires_grad_`'s main use case is to tell autograd to begin recording
 |      operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
 |      (because it was obtained through a DataLoader, or required preprocessing or
 |      initialization), ``tensor.requires_grad_()`` makes it so that autograd will
 |      begin to record operations on ``tensor``.
 |      
 |      Args:
 |          requires_grad (bool): If autograd should record operations on this tensor.
 |              Default: ``True``.
 |      
 |      Example::
 |      
 |          >>> # Let's say we want to preprocess some saved weights and use
 |          >>> # the result as new weights.
 |          >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
 |          >>> loaded_weights = torch.tensor(saved_weights)
 |          >>> weights = preprocess(loaded_weights)  # some function
 |          >>> weights
 |          tensor([-0.5503,  0.4926, -2.1158, -0.8303])
 |      
 |          >>> # Now, start to record operations done to weights
 |          >>> weights.requires_grad_()
 |          >>> out = weights.pow(2).sum()
 |          >>> out.backward()
 |          >>> weights.grad
 |          tensor([-1.1007,  0.9853, -4.2316, -1.6606])
 |  
 |  rreesshhaappee(...)
 |      reshape(*shape) -> Tensor
 |      
 |      Returns a tensor with the same data and number of elements as :attr:`self`
 |      but with the specified shape. This method returns a view if :attr:`shape` is
 |      compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
 |      possible to return a view.
 |      
 |      See :func:`torch.reshape`
 |      
 |      Args:
 |          shape (tuple of ints or int...): the desired shape
 |  
 |  rreesshhaappee__aass(...)
 |      reshape_as(other) -> Tensor
 |      
 |      Returns this tensor as the same shape as :attr:`other`.
 |      ``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.
 |      This method returns a view if ``other.sizes()`` is compatible with the current
 |      shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.
 |      
 |      Please see :meth:`reshape` for more information about ``reshape``.
 |      
 |      Args:
 |          other (:class:`torch.Tensor`): The result tensor has the same shape
 |              as :attr:`other`.
 |  
 |  rreessiizzee__(...)
 |      resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor
 |      
 |      Resizes :attr:`self` tensor to the specified size. If the number of elements is
 |      larger than the current storage size, then the underlying storage is resized
 |      to fit the new number of elements. If the number of elements is smaller, the
 |      underlying storage is not changed. Existing elements are preserved but any new
 |      memory is uninitialized.
 |      
 |      .. warning::
 |      
 |          This is a low-level method. The storage is reinterpreted as C-contiguous,
 |          ignoring the current strides (unless the target size equals the current
 |          size, in which case the tensor is left unchanged). For most purposes, you
 |          will instead want to use :meth:`~Tensor.view()`, which checks for
 |          contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
 |          change the size in-place with custom strides, see :meth:`~Tensor.set_()`.
 |      
 |      Args:
 |          sizes (torch.Size or int...): the desired size
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              Tensor. Default: ``torch.contiguous_format``. Note that memory format of
 |              :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
 |          >>> x.resize_(2, 2)
 |          tensor([[ 1,  2],
 |                  [ 3,  4]])
 |  
 |  rreessiizzee__aass__(...)
 |      resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor
 |      
 |      Resizes the :attr:`self` tensor to be the same size as the specified
 |      :attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              Tensor. Default: ``torch.contiguous_format``. Note that memory format of
 |              :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.
 |  
 |  rroollll(...)
 |      roll(shifts, dims) -> Tensor
 |      
 |      See :func:`torch.roll`
 |  
 |  rroott9900(...)
 |      rot90(k, dims) -> Tensor
 |      
 |      See :func:`torch.rot90`
 |  
 |  rroouunndd(...)
 |      round() -> Tensor
 |      
 |      See :func:`torch.round`
 |  
 |  rroouunndd__(...)
 |      round_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.round`
 |  
 |  rrssqqrrtt(...)
 |      rsqrt() -> Tensor
 |      
 |      See :func:`torch.rsqrt`
 |  
 |  rrssqqrrtt__(...)
 |      rsqrt_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.rsqrt`
 |  
 |  ssccaatttteerr(...)
 |      scatter(dim, index, src) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.scatter_`
 |  
 |  ssccaatttteerr__(...)
 |      scatter_(dim, index, src, reduce=None) -> Tensor
 |      
 |      Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
 |      specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
 |      index is specified by its index in :attr:`src` for ``dimension != dim`` and by
 |      the corresponding value in :attr:`index` for ``dimension = dim``.
 |      
 |      For a 3-D tensor, :attr:`self` is updated as::
 |      
 |          self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
 |          self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
 |          self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
 |      
 |      This is the reverse operation of the manner described in :meth:`~Tensor.gather`.
 |      
 |      :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
 |      the same number of dimensions. It is also required that
 |      ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
 |      ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
 |      Note that ``index`` and ``src`` do not broadcast.
 |      
 |      Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
 |      between ``0`` and ``self.size(dim) - 1`` inclusive.
 |      
 |      .. warning::
 |      
 |          When indices are not unique, the behavior is non-deterministic (one of the
 |          values from ``src`` will be picked arbitrarily) and the gradient will be
 |          incorrect (it will be propagated to all locations in the source that
 |          correspond to the same index)!
 |      
 |      .. note::
 |      
 |          The backward pass is implemented only for ``src.shape == index.shape``.
 |      
 |      Additionally accepts an optional :attr:`reduce` argument that allows
 |      specification of an optional reduction operation, which is applied to all
 |      values in the tensor :attr:`src` into :attr:`self` at the indicies
 |      specified in the :attr:`index`. For each value in :attr:`src`, the reduction
 |      operation is applied to an index in :attr:`self` which is specified by
 |      its index in :attr:`src` for ``dimension != dim`` and by the corresponding
 |      value in :attr:`index` for ``dimension = dim``.
 |      
 |      Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
 |      is updated as::
 |      
 |          self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
 |          self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
 |          self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2
 |      
 |      Reducing with the addition operation is the same as using
 |      :meth:`~torch.Tensor.scatter_add_`.
 |      
 |      Args:
 |          dim (int): the axis along which to index
 |          index (LongTensor): the indices of elements to scatter, can be either empty
 |              or of the same dimensionality as ``src``. When empty, the operation
 |              returns ``self`` unchanged.
 |          src (Tensor or float): the source element(s) to scatter.
 |          reduce (str, optional): reduction operation to apply, can be either
 |              ``'add'`` or ``'multiply'``.
 |      
 |      Example::
 |      
 |          >>> src = torch.arange(1, 11).reshape((2, 5))
 |          >>> src
 |          tensor([[ 1,  2,  3,  4,  5],
 |                  [ 6,  7,  8,  9, 10]])
 |          >>> index = torch.tensor([[0, 1, 2, 0]])
 |          >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
 |          tensor([[1, 0, 0, 4, 0],
 |                  [0, 2, 0, 0, 0],
 |                  [0, 0, 3, 0, 0]])
 |          >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
 |          >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
 |          tensor([[1, 2, 3, 0, 0],
 |                  [6, 7, 0, 0, 8],
 |                  [0, 0, 0, 0, 0]])
 |      
 |          >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
 |          ...            1.23, reduce='multiply')
 |          tensor([[2.0000, 2.0000, 2.4600, 2.0000],
 |                  [2.0000, 2.0000, 2.0000, 2.4600]])
 |          >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
 |          ...            1.23, reduce='add')
 |          tensor([[2.0000, 2.0000, 3.2300, 2.0000],
 |                  [2.0000, 2.0000, 2.0000, 3.2300]])
 |  
 |  ssccaatttteerr__aadddd(...)
 |      scatter_add(dim, index, src) -> Tensor
 |      
 |      Out-of-place version of :meth:`torch.Tensor.scatter_add_`
 |  
 |  ssccaatttteerr__aadddd__(...)
 |      scatter_add_(dim, index, src) -> Tensor
 |      
 |      Adds all values from the tensor :attr:`other` into :attr:`self` at the indices
 |      specified in the :attr:`index` tensor in a similar fashion as
 |      :meth:`~torch.Tensor.scatter_`. For each value in :attr:`src`, it is added to
 |      an index in :attr:`self` which is specified by its index in :attr:`src`
 |      for ``dimension != dim`` and by the corresponding value in :attr:`index` for
 |      ``dimension = dim``.
 |      
 |      For a 3-D tensor, :attr:`self` is updated as::
 |      
 |          self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
 |          self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
 |          self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
 |      
 |      :attr:`self`, :attr:`index` and :attr:`src` should have same number of
 |      dimensions. It is also required that ``index.size(d) <= src.size(d)`` for all
 |      dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all dimensions
 |      ``d != dim``. Note that ``index`` and ``src`` do not broadcast.
 |      
 |      Note:
 |          This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.
 |      
 |      .. note::
 |      
 |          The backward pass is implemented only for ``src.shape == index.shape``.
 |      
 |      Args:
 |          dim (int): the axis along which to index
 |          index (LongTensor): the indices of elements to scatter and add, can be
 |              either empty or of the same dimensionality as ``src``. When empty, the
 |              operation returns ``self`` unchanged.
 |          src (Tensor): the source elements to scatter and add
 |      
 |      Example::
 |      
 |          >>> src = torch.ones((2, 5))
 |          >>> index = torch.tensor([[0, 1, 2, 0, 0]])
 |          >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
 |          tensor([[1., 0., 0., 1., 1.],
 |                  [0., 1., 0., 0., 0.],
 |                  [0., 0., 1., 0., 0.]])
 |          >>> index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
 |          >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
 |          tensor([[2., 0., 0., 1., 1.],
 |                  [0., 2., 0., 0., 0.],
 |                  [0., 0., 2., 1., 1.]])
 |  
 |  sseelleecctt(...)
 |      select(dim, index) -> Tensor
 |      
 |      Slices the :attr:`self` tensor along the selected dimension at the given index.
 |      This function returns a view of the original tensor with the given dimension removed.
 |      
 |      Args:
 |          dim (int): the dimension to slice
 |          index (int): the index to select with
 |      
 |      .. note::
 |      
 |          :meth:`select` is equivalent to slicing. For example,
 |          ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and
 |          ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.
 |  
 |  sseett__(...)
 |      set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor
 |      
 |      Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
 |      :attr:`self` tensor will share the same storage and have the same size and
 |      strides as :attr:`source`. Changes to elements in one tensor will be reflected
 |      in the other.
 |      
 |      If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
 |      storage, offset, size, and stride.
 |      
 |      Args:
 |          source (Tensor or Storage): the tensor or storage to use
 |          storage_offset (int, optional): the offset in the storage
 |          size (torch.Size, optional): the desired size. Defaults to the size of the source.
 |          stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
 |  
 |  ssggnn(...)
 |      sgn() -> Tensor
 |      
 |      See :func:`torch.sgn`
 |  
 |  ssggnn__(...)
 |      sgn_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sgn`
 |  
 |  sshhoorrtt(...)
 |      short(memory_format=torch.preserve_format) -> Tensor
 |      
 |      ``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.
 |      
 |      Args:
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  ssiiggmmooiidd(...)
 |      sigmoid() -> Tensor
 |      
 |      See :func:`torch.sigmoid`
 |  
 |  ssiiggmmooiidd__(...)
 |      sigmoid_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sigmoid`
 |  
 |  ssiiggnn(...)
 |      sign() -> Tensor
 |      
 |      See :func:`torch.sign`
 |  
 |  ssiiggnn__(...)
 |      sign_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sign`
 |  
 |  ssiiggnnbbiitt(...)
 |      signbit() -> Tensor
 |      
 |      See :func:`torch.signbit`
 |  
 |  ssiinn(...)
 |      sin() -> Tensor
 |      
 |      See :func:`torch.sin`
 |  
 |  ssiinn__(...)
 |      sin_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sin`
 |  
 |  ssiinncc(...)
 |      sinc() -> Tensor
 |      
 |      See :func:`torch.sinc`
 |  
 |  ssiinncc__(...)
 |      sinc_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sinc`
 |  
 |  ssiinnhh(...)
 |      sinh() -> Tensor
 |      
 |      See :func:`torch.sinh`
 |  
 |  ssiinnhh__(...)
 |      sinh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sinh`
 |  
 |  ssiizzee(...)
 |      size() -> torch.Size
 |      
 |      Returns the size of the :attr:`self` tensor. The returned value is a subclass of
 |      :class:`tuple`.
 |      
 |      Example::
 |      
 |          >>> torch.empty(3, 4, 5).size()
 |          torch.Size([3, 4, 5])
 |  
 |  ssllooggddeett(...)
 |      slogdet() -> (Tensor, Tensor)
 |      
 |      See :func:`torch.slogdet`
 |  
 |  ssmmmm(...)
 |      smm(mat) -> Tensor
 |      
 |      See :func:`torch.smm`
 |  
 |  ssooffttmmaaxx(...)
 |  
 |  ssoollvvee(...)
 |      solve(A) -> Tensor, Tensor
 |      
 |      See :func:`torch.solve`
 |  
 |  ssoorrtt(...)
 |      sort(dim=-1, descending=False) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.sort`
 |  
 |  ssppaarrssee__ddiimm(...)
 |      sparse_dim() -> int
 |      
 |      Return the number of sparse dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse tensor.
 |      
 |      See also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
 |  
 |  ssppaarrssee__mmaasskk(...)
 |      sparse_mask(mask) -> Tensor
 |      
 |      Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
 |      strided tensor :attr:`self` filtered by the indices of the sparse
 |      tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
 |      ignored. :attr:`self` and :attr:`mask` tensors must have the same
 |      shape.
 |      
 |      .. note::
 |      
 |        The returned sparse tensor has the same indices as the sparse tensor
 |        :attr:`mask`, even when the corresponding values in :attr:`self` are
 |        zeros.
 |      
 |      Args:
 |          mask (Tensor): a sparse tensor whose indices are used as a filter
 |      
 |      Example::
 |      
 |          >>> nse = 5
 |          >>> dims = (5, 5, 2, 2)
 |          >>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
 |          ...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
 |          >>> V = torch.randn(nse, dims[2], dims[3])
 |          >>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
 |          >>> D = torch.randn(dims)
 |          >>> D.sparse_mask(S)
 |          tensor(indices=tensor([[0, 0, 0, 2],
 |                                 [0, 1, 4, 3]]),
 |                 values=tensor([[[ 1.6550,  0.2397],
 |                                 [-0.1611, -0.0779]],
 |      
 |                                [[ 0.2326, -1.0558],
 |                                 [ 1.4711,  1.9678]],
 |      
 |                                [[-0.5138, -0.0411],
 |                                 [ 1.9417,  0.5158]],
 |      
 |                                [[ 0.0793,  0.0036],
 |                                 [-0.2569, -0.1055]]]),
 |                 size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
 |  
 |  ssppaarrssee__rreessiizzee__(...)
 |      sparse_resize_(size, sparse_dim, dense_dim) -> Tensor
 |      
 |      Resizes :attr:`self` :ref:`sparse tensor <sparse-docs>` to the desired
 |      size and the number of sparse and dense dimensions.
 |      
 |      .. note::
 |        If the number of specified elements in :attr:`self` is zero, then
 |        :attr:`size`, :attr:`sparse_dim`, and :attr:`dense_dim` can be any
 |        size and positive integers such that ``len(size) == sparse_dim +
 |        dense_dim``.
 |      
 |        If :attr:`self` specifies one or more elements, however, then each
 |        dimension in :attr:`size` must not be smaller than the corresponding
 |        dimension of :attr:`self`, :attr:`sparse_dim` must equal the number
 |        of sparse dimensions in :attr:`self`, and :attr:`dense_dim` must
 |        equal the number of dense dimensions in :attr:`self`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse tensor.
 |      
 |      Args:
 |          size (torch.Size): the desired size. If :attr:`self` is non-empty
 |            sparse tensor, the desired size cannot be smaller than the
 |            original size.
 |          sparse_dim (int): the number of sparse dimensions
 |          dense_dim (int): the number of dense dimensions
 |  
 |  ssppaarrssee__rreessiizzee__aanndd__cclleeaarr__(...)
 |      sparse_resize_and_clear_(size, sparse_dim, dense_dim) -> Tensor
 |      
 |      Removes all specified elements from a :ref:`sparse tensor
 |      <sparse-docs>` :attr:`self` and resizes :attr:`self` to the desired
 |      size and the number of sparse and dense dimensions.
 |      
 |      .. warning:
 |        Throws an error if :attr:`self` is not a sparse tensor.
 |      
 |      Args:
 |          size (torch.Size): the desired size.
 |          sparse_dim (int): the number of sparse dimensions
 |          dense_dim (int): the number of dense dimensions
 |  
 |  sspplliitt__wwiitthh__ssiizzeess(...)
 |  
 |  ssqqrrtt(...)
 |      sqrt() -> Tensor
 |      
 |      See :func:`torch.sqrt`
 |  
 |  ssqqrrtt__(...)
 |      sqrt_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sqrt`
 |  
 |  ssqquuaarree(...)
 |      square() -> Tensor
 |      
 |      See :func:`torch.square`
 |  
 |  ssqquuaarree__(...)
 |      square_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.square`
 |  
 |  ssqquueeeezzee(...)
 |      squeeze(dim=None) -> Tensor
 |      
 |      See :func:`torch.squeeze`
 |  
 |  ssqquueeeezzee__(...)
 |      squeeze_(dim=None) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.squeeze`
 |  
 |  ssssppaaddddmmmm(...)
 |      sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor
 |      
 |      See :func:`torch.sspaddmm`
 |  
 |  ssttdd(...)
 |      std(dim=None, unbiased=True, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.std`
 |  
 |  ssttoorraaggee(...)
 |      storage() -> torch.Storage
 |      
 |      Returns the underlying storage.
 |  
 |  ssttoorraaggee__ooffffsseett(...)
 |      storage_offset() -> int
 |      
 |      Returns :attr:`self` tensor's offset in the underlying storage in terms of
 |      number of storage elements (not bytes).
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([1, 2, 3, 4, 5])
 |          >>> x.storage_offset()
 |          0
 |          >>> x[3:].storage_offset()
 |          3
 |  
 |  ssttoorraaggee__ttyyppee(...)
 |      storage_type() -> type
 |      
 |      Returns the type of the underlying storage.
 |  
 |  ssttrriiddee(...)
 |      stride(dim) -> tuple or int
 |      
 |      Returns the stride of :attr:`self` tensor.
 |      
 |      Stride is the jump necessary to go from one element to the next one in the
 |      specified dimension :attr:`dim`. A tuple of all strides is returned when no
 |      argument is passed in. Otherwise, an integer value is returned as the stride in
 |      the particular dimension :attr:`dim`.
 |      
 |      Args:
 |          dim (int, optional): the desired dimension in which stride is required
 |      
 |      Example::
 |      
 |          >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
 |          >>> x.stride()
 |          (5, 1)
 |          >>> x.stride(0)
 |          5
 |          >>> x.stride(-1)
 |          1
 |  
 |  ssuubb(...)
 |      sub(other, *, alpha=1) -> Tensor
 |      
 |      See :func:`torch.sub`.
 |  
 |  ssuubb__(...)
 |      sub_(other, *, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.sub`
 |  
 |  ssuubbttrraacctt(...)
 |      subtract(other, *, alpha=1) -> Tensor
 |      
 |      See :func:`torch.subtract`.
 |  
 |  ssuubbttrraacctt__(...)
 |      subtract_(other, *, alpha=1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.subtract`.
 |  
 |  ssuumm(...)
 |      sum(dim=None, keepdim=False, dtype=None) -> Tensor
 |      
 |      See :func:`torch.sum`
 |  
 |  ssuumm__ttoo__ssiizzee(...)
 |      sum_to_size(*size) -> Tensor
 |      
 |      Sum ``this`` tensor to :attr:`size`.
 |      :attr:`size` must be broadcastable to ``this`` tensor size.
 |      
 |      Args:
 |          size (int...): a sequence of integers defining the shape of the output tensor.
 |  
 |  ssvvdd(...)
 |      svd(some=True, compute_uv=True) -> (Tensor, Tensor, Tensor)
 |      
 |      See :func:`torch.svd`
 |  
 |  sswwaappaaxxeess(...)
 |      swapaxes(axis0, axis1) -> Tensor
 |      
 |      See :func:`torch.swapaxes`
 |  
 |  sswwaappaaxxeess__(...)
 |      swapaxes_(axis0, axis1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.swapaxes`
 |  
 |  sswwaappddiimmss(...)
 |      swapdims(dim0, dim1) -> Tensor
 |      
 |      See :func:`torch.swapdims`
 |  
 |  sswwaappddiimmss__(...)
 |      swapdims_(dim0, dim1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.swapdims`
 |  
 |  ssyymmeeiigg(...)
 |      symeig(eigenvectors=False, upper=True) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.symeig`
 |  
 |  tt(...)
 |      t() -> Tensor
 |      
 |      See :func:`torch.t`
 |  
 |  tt__(...)
 |      t_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.t`
 |  
 |  ttaakkee(...)
 |      take(indices) -> Tensor
 |      
 |      See :func:`torch.take`
 |  
 |  ttaann(...)
 |      tan() -> Tensor
 |      
 |      See :func:`torch.tan`
 |  
 |  ttaann__(...)
 |      tan_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.tan`
 |  
 |  ttaannhh(...)
 |      tanh() -> Tensor
 |      
 |      See :func:`torch.tanh`
 |  
 |  ttaannhh__(...)
 |      tanh_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.tanh`
 |  
 |  tteennssoorr__sspplliitt(...)
 |      tensor_split(indices_or_sections, dim=0) -> List of Tensors
 |      
 |      See :func:`torch.tensor_split`
 |  
 |  ttiillee(...)
 |      tile(*reps) -> Tensor
 |      
 |      See :func:`torch.tile`
 |  
 |  ttoo(...)
 |      to(*args, **kwargs) -> Tensor
 |      
 |      Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
 |      inferred from the arguments of ``self.to(*args, **kwargs)``.
 |      
 |      .. note::
 |      
 |          If the ``self`` Tensor already
 |          has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
 |          Otherwise, the returned tensor is a copy of ``self`` with the desired
 |          :class:`torch.dtype` and :class:`torch.device`.
 |      
 |      Here are the ways to call ``to``:
 |      
 |      .. function:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
 |      
 |          Returns a Tensor with the specified :attr:`dtype`
 |      
 |          Args:
 |              memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |      
 |      .. function:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
 |      
 |          Returns a Tensor with the specified :attr:`device` and (optional)
 |          :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
 |          When :attr:`non_blocking`, tries to convert asynchronously with respect to
 |          the host if possible, e.g., converting a CPU Tensor with pinned memory to a
 |          CUDA Tensor.
 |          When :attr:`copy` is set, a new Tensor is created even when the Tensor
 |          already matches the desired conversion.
 |      
 |          Args:
 |              memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |      
 |      .. function:: to(other, non_blocking=False, copy=False) -> Tensor
 |      
 |          Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
 |          the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
 |          asynchronously with respect to the host if possible, e.g., converting a CPU
 |          Tensor with pinned memory to a CUDA Tensor.
 |          When :attr:`copy` is set, a new Tensor is created even when the Tensor
 |          already matches the desired conversion.
 |      
 |      Example::
 |      
 |          >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
 |          >>> tensor.to(torch.float64)
 |          tensor([[-0.5044,  0.0005],
 |                  [ 0.3310, -0.0584]], dtype=torch.float64)
 |      
 |          >>> cuda0 = torch.device('cuda:0')
 |          >>> tensor.to(cuda0)
 |          tensor([[-0.5044,  0.0005],
 |                  [ 0.3310, -0.0584]], device='cuda:0')
 |      
 |          >>> tensor.to(cuda0, dtype=torch.float64)
 |          tensor([[-0.5044,  0.0005],
 |                  [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
 |      
 |          >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
 |          >>> tensor.to(other, non_blocking=True)
 |          tensor([[-0.5044,  0.0005],
 |                  [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
 |  
 |  ttoo__ddeennssee(...)
 |      to_dense() -> Tensor
 |      
 |      Creates a strided copy of :attr:`self`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is a strided tensor.
 |      
 |      Example::
 |      
 |          >>> s = torch.sparse_coo_tensor(
 |          ...        torch.tensor([[1, 1],
 |          ...                      [0, 2]]),
 |          ...        torch.tensor([9, 10]),
 |          ...        size=(3, 3))
 |          >>> s.to_dense()
 |          tensor([[ 0,  0,  0],
 |                  [ 9,  0, 10],
 |                  [ 0,  0,  0]])
 |  
 |  ttoo__mmkkllddnnnn(...)
 |      to_mkldnn() -> Tensor
 |      Returns a copy of the tensor in ``torch.mkldnn`` layout.
 |  
 |  ttoo__ssppaarrssee(...)
 |      to_sparse(sparseDims) -> Tensor
 |      Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
 |      :ref:`coordinate format <sparse-coo-docs>`.
 |      
 |      Args:
 |          sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor
 |      
 |      Example::
 |      
 |          >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
 |          >>> d
 |          tensor([[ 0,  0,  0],
 |                  [ 9,  0, 10],
 |                  [ 0,  0,  0]])
 |          >>> d.to_sparse()
 |          tensor(indices=tensor([[1, 1],
 |                                 [0, 2]]),
 |                 values=tensor([ 9, 10]),
 |                 size=(3, 3), nnz=2, layout=torch.sparse_coo)
 |          >>> d.to_sparse(1)
 |          tensor(indices=tensor([[1]]),
 |                 values=tensor([[ 9,  0, 10]]),
 |                 size=(3, 3), nnz=1, layout=torch.sparse_coo)
 |  
 |  ttoolliisstt(...)
 |      tolist() -> list or number
 |      
 |      Returns the tensor as a (nested) list. For scalars, a standard
 |      Python number is returned, just like with :meth:`~Tensor.item`.
 |      Tensors are automatically moved to the CPU first if necessary.
 |      
 |      This operation is not differentiable.
 |      
 |      Examples::
 |      
 |          >>> a = torch.randn(2, 2)
 |          >>> a.tolist()
 |          [[0.012766935862600803, 0.5415473580360413],
 |           [-0.08909505605697632, 0.7729271650314331]]
 |          >>> a[0,0].tolist()
 |          0.012766935862600803
 |  
 |  ttooppkk(...)
 |      topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)
 |      
 |      See :func:`torch.topk`
 |  
 |  ttrraaccee(...)
 |      trace() -> Tensor
 |      
 |      See :func:`torch.trace`
 |  
 |  ttrraannssppoossee(...)
 |      transpose(dim0, dim1) -> Tensor
 |      
 |      See :func:`torch.transpose`
 |  
 |  ttrraannssppoossee__(...)
 |      transpose_(dim0, dim1) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.transpose`
 |  
 |  ttrriiaanngguullaarr__ssoollvvee(...)
 |      triangular_solve(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
 |      
 |      See :func:`torch.triangular_solve`
 |  
 |  ttrriill(...)
 |      tril(k=0) -> Tensor
 |      
 |      See :func:`torch.tril`
 |  
 |  ttrriill__(...)
 |      tril_(k=0) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.tril`
 |  
 |  ttrriiuu(...)
 |      triu(k=0) -> Tensor
 |      
 |      See :func:`torch.triu`
 |  
 |  ttrriiuu__(...)
 |      triu_(k=0) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.triu`
 |  
 |  ttrruuee__ddiivviiddee(...)
 |      true_divide(value) -> Tensor
 |      
 |      See :func:`torch.true_divide`
 |  
 |  ttrruuee__ddiivviiddee__(...)
 |      true_divide_(value) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.true_divide_`
 |  
 |  ttrruunncc(...)
 |      trunc() -> Tensor
 |      
 |      See :func:`torch.trunc`
 |  
 |  ttrruunncc__(...)
 |      trunc_() -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.trunc`
 |  
 |  ttyyppee(...)
 |      type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
 |      Returns the type if `dtype` is not provided, else casts this object to
 |      the specified type.
 |      
 |      If this is already of the correct type, no copy is performed and the
 |      original object is returned.
 |      
 |      Args:
 |          dtype (type or string): The desired type
 |          non_blocking (bool): If ``True``, and the source is in pinned memory
 |              and destination is on the GPU or vice versa, the copy is performed
 |              asynchronously with respect to the host. Otherwise, the argument
 |              has no effect.
 |          **kwargs: For compatibility, may contain the key ``async`` in place of
 |              the ``non_blocking`` argument. The ``async`` arg is deprecated.
 |  
 |  ttyyppee__aass(...)
 |      type_as(tensor) -> Tensor
 |      
 |      Returns this tensor cast to the type of the given tensor.
 |      
 |      This is a no-op if the tensor is already of the correct type. This is
 |      equivalent to ``self.type(tensor.type())``
 |      
 |      Args:
 |          tensor (Tensor): the tensor which has the desired type
 |  
 |  uunnbbiinndd(...)
 |      unbind(dim=0) -> seq
 |      
 |      See :func:`torch.unbind`
 |  
 |  uunnffoolldd(...)
 |      unfold(dimension, size, step) -> Tensor
 |      
 |      Returns a view of the original tensor which contains all slices of size :attr:`size` from
 |      :attr:`self` tensor in the dimension :attr:`dimension`.
 |      
 |      Step between two slices is given by :attr:`step`.
 |      
 |      If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
 |      dimension :attr:`dimension` in the returned tensor will be
 |      `(sizedim - size) / step + 1`.
 |      
 |      An additional dimension of size :attr:`size` is appended in the returned tensor.
 |      
 |      Args:
 |          dimension (int): dimension in which unfolding happens
 |          size (int): the size of each slice that is unfolded
 |          step (int): the step between each slice
 |      
 |      Example::
 |      
 |          >>> x = torch.arange(1., 8)
 |          >>> x
 |          tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
 |          >>> x.unfold(0, 2, 1)
 |          tensor([[ 1.,  2.],
 |                  [ 2.,  3.],
 |                  [ 3.,  4.],
 |                  [ 4.,  5.],
 |                  [ 5.,  6.],
 |                  [ 6.,  7.]])
 |          >>> x.unfold(0, 2, 2)
 |          tensor([[ 1.,  2.],
 |                  [ 3.,  4.],
 |                  [ 5.,  6.]])
 |  
 |  uunniiffoorrmm__(...)
 |      uniform_(from=0, to=1) -> Tensor
 |      
 |      Fills :attr:`self` tensor with numbers sampled from the continuous uniform
 |      distribution:
 |      
 |      .. math::
 |          P(x) = \dfrac{1}{\text{to} - \text{from}}
 |  
 |  uunnssaaffee__cchhuunnkk(...)
 |      unsafe_chunk(chunks, dim=0) -> List of Tensors
 |      
 |      See :func:`torch.unsafe_chunk`
 |  
 |  uunnssaaffee__sspplliitt(...)
 |      unsafe_split(split_size, dim=0) -> List of Tensors
 |      
 |      See :func:`torch.unsafe_split`
 |  
 |  uunnssaaffee__sspplliitt__wwiitthh__ssiizzeess(...)
 |  
 |  uunnssqquueeeezzee(...)
 |      unsqueeze(dim) -> Tensor
 |      
 |      See :func:`torch.unsqueeze`
 |  
 |  uunnssqquueeeezzee__(...)
 |      unsqueeze_(dim) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.unsqueeze`
 |  
 |  vvaalluueess(...)
 |      values() -> Tensor
 |      
 |      Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.
 |      
 |      .. warning::
 |        Throws an error if :attr:`self` is not a sparse COO tensor.
 |      
 |      See also :meth:`Tensor.indices`.
 |      
 |      .. note::
 |        This method can only be called on a coalesced sparse tensor. See
 |        :meth:`Tensor.coalesce` for details.
 |  
 |  vvaarr(...)
 |      var(dim=None, unbiased=True, keepdim=False) -> Tensor
 |      
 |      See :func:`torch.var`
 |  
 |  vvddoott(...)
 |      vdot(other) -> Tensor
 |      
 |      See :func:`torch.vdot`
 |  
 |  vviieeww(...)
 |      view(*shape) -> Tensor
 |      
 |      Returns a new tensor with the same data as the :attr:`self` tensor but of a
 |      different :attr:`shape`.
 |      
 |      The returned tensor shares the same data and must have the same number
 |      of elements, but may have a different size. For a tensor to be viewed, the new
 |      view size must be compatible with its original size and stride, i.e., each new
 |      view dimension must either be a subspace of an original dimension, or only span
 |      across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
 |      contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,
 |      
 |      .. math::
 |      
 |        \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
 |      
 |      Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
 |      without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
 |      :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
 |      returns a view if the shapes are compatible, and copies (equivalent to calling
 |      :meth:`contiguous`) otherwise.
 |      
 |      Args:
 |          shape (torch.Size or int...): the desired size
 |      
 |      Example::
 |      
 |          >>> x = torch.randn(4, 4)
 |          >>> x.size()
 |          torch.Size([4, 4])
 |          >>> y = x.view(16)
 |          >>> y.size()
 |          torch.Size([16])
 |          >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
 |          >>> z.size()
 |          torch.Size([2, 8])
 |      
 |          >>> a = torch.randn(1, 2, 3, 4)
 |          >>> a.size()
 |          torch.Size([1, 2, 3, 4])
 |          >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
 |          >>> b.size()
 |          torch.Size([1, 3, 2, 4])
 |          >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
 |          >>> c.size()
 |          torch.Size([1, 3, 2, 4])
 |          >>> torch.equal(b, c)
 |          False
 |      
 |      
 |      .. function:: view(dtype) -> Tensor
 |      
 |      Returns a new tensor with the same data as the :attr:`self` tensor but of a
 |      different :attr:`dtype`. :attr:`dtype` must have the same number of bytes per
 |      element as :attr:`self`'s dtype.
 |      
 |      .. warning::
 |      
 |          This overload is not supported by TorchScript, and using it in a Torchscript
 |          program will cause undefined behavior.
 |      
 |      
 |      Args:
 |          dtype (:class:`torch.dtype`): the desired dtype
 |      
 |      Example::
 |      
 |          >>> x = torch.randn(4, 4)
 |          >>> x
 |          tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
 |                  [-0.1520,  0.7472,  0.5617, -0.8649],
 |                  [-2.4724, -0.0334, -0.2976, -0.8499],
 |                  [-0.2109,  1.9913, -0.9607, -0.6123]])
 |          >>> x.dtype
 |          torch.float32
 |      
 |          >>> y = x.view(torch.int32)
 |          >>> y
 |          tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
 |                  [-1105482831,  1061112040,  1057999968, -1084397505],
 |                  [-1071760287, -1123489973, -1097310419, -1084649136],
 |                  [-1101533110,  1073668768, -1082790149, -1088634448]],
 |              dtype=torch.int32)
 |          >>> y[0, 0] = 1000000000
 |          >>> x
 |          tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
 |                  [-0.1520,  0.7472,  0.5617, -0.8649],
 |                  [-2.4724, -0.0334, -0.2976, -0.8499],
 |                  [-0.2109,  1.9913, -0.9607, -0.6123]])
 |      
 |          >>> x.view(torch.int16)
 |          Traceback (most recent call last):
 |            File "<stdin>", line 1, in <module>
 |          RuntimeError: Viewing a tensor as a new dtype with a different number of bytes per element is not supported.
 |  
 |  vviieeww__aass(...)
 |      view_as(other) -> Tensor
 |      
 |      View this tensor as the same size as :attr:`other`.
 |      ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.
 |      
 |      Please see :meth:`~Tensor.view` for more information about ``view``.
 |      
 |      Args:
 |          other (:class:`torch.Tensor`): The result tensor has the same size
 |              as :attr:`other`.
 |  
 |  wwhheerree(...)
 |      where(condition, y) -> Tensor
 |      
 |      ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
 |      See :func:`torch.where`
 |  
 |  xxllooggyy(...)
 |      xlogy(other) -> Tensor
 |      
 |      See :func:`torch.xlogy`
 |  
 |  xxllooggyy__(...)
 |      xlogy_(other) -> Tensor
 |      
 |      In-place version of :meth:`~Tensor.xlogy`
 |  
 |  xxppuu(...)
 |      xpu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor
 |      
 |      Returns a copy of this object in XPU memory.
 |      
 |      If this object is already in XPU memory and on the correct device,
 |      then no copy is performed and the original object is returned.
 |      
 |      Args:
 |          device (:class:`torch.device`): The destination XPU device.
 |              Defaults to the current XPU device.
 |          non_blocking (bool): If ``True`` and the source is in pinned memory,
 |              the copy will be asynchronous with respect to the host.
 |              Otherwise, the argument has no effect. Default: ``False``.
 |          memory_format (:class:`torch.memory_format`, optional): the desired memory format of
 |              returned Tensor. Default: ``torch.preserve_format``.
 |  
 |  zzeerroo__(...)
 |      zero_() -> Tensor
 |      
 |      Fills :attr:`self` tensor with zeros.
 |  
 |  ----------------------------------------------------------------------
 |  Static methods inherited from torch._C._TensorBase:
 |  
 |  ____nneeww____(*args, **kwargs) from builtins.type
 |      Create and return a new object.  See help(type) for accurate signature.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from torch._C._TensorBase:
 |  
 |  TT
 |      Is this Tensor with its dimensions reversed.
 |      
 |      If ``n`` is the number of dimensions in ``x``,
 |      ``x.T`` is equivalent to ``x.permute(n-1, n-2, ..., 0)``.
 |  
 |  ddaattaa
 |  
 |  ddeevviiccee
 |      Is the :class:`torch.device` where this Tensor is.
 |  
 |  ddttyyppee
 |  
 |  ggrraadd__ffnn
 |  
 |  iimmaagg
 |      Returns a new tensor containing imaginary values of the :attr:`self` tensor.
 |      The returned tensor and :attr:`self` share the same underlying storage.
 |      
 |      .. warning::
 |          :func:`imag` is only supported for tensors with complex dtypes.
 |      
 |      Example::
 |          >>> x=torch.randn(4, dtype=torch.cfloat)
 |          >>> x
 |          tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
 |          >>> x.imag
 |          tensor([ 0.3553, -0.7896, -0.0633, -0.8119])
 |  
 |  iiss__ccuuddaa
 |      Is ``True`` if the Tensor is stored on the GPU, ``False`` otherwise.
 |  
 |  iiss__lleeaaff
 |      All Tensors that have :attr:`requires_grad` which is ``False`` will be leaf Tensors by convention.
 |      
 |      For Tensors that have :attr:`requires_grad` which is ``True``, they will be leaf Tensors if they were
 |      created by the user. This means that they are not the result of an operation and so
 |      :attr:`grad_fn` is None.
 |      
 |      Only leaf Tensors will have their :attr:`grad` populated during a call to :func:`backward`.
 |      To get :attr:`grad` populated for non-leaf Tensors, you can use :func:`retain_grad`.
 |      
 |      Example::
 |      
 |          >>> a = torch.rand(10, requires_grad=True)
 |          >>> a.is_leaf
 |          True
 |          >>> b = torch.rand(10, requires_grad=True).cuda()
 |          >>> b.is_leaf
 |          False
 |          # b was created by the operation that cast a cpu Tensor into a cuda Tensor
 |          >>> c = torch.rand(10, requires_grad=True) + 2
 |          >>> c.is_leaf
 |          False
 |          # c was created by the addition operation
 |          >>> d = torch.rand(10).cuda()
 |          >>> d.is_leaf
 |          True
 |          # d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
 |          >>> e = torch.rand(10).cuda().requires_grad_()
 |          >>> e.is_leaf
 |          True
 |          # e requires gradients and has no operations creating it
 |          >>> f = torch.rand(10, requires_grad=True, device="cuda")
 |          >>> f.is_leaf
 |          True
 |          # f requires grad, has no operation creating it
 |  
 |  iiss__mmeettaa
 |      Is ``True`` if the Tensor is a meta tensor, ``False`` otherwise.  Meta tensors
 |      are like normal tensors, but they carry no data.
 |  
 |  iiss__mmkkllddnnnn
 |  
 |  iiss__qquuaannttiizzeedd
 |      Is ``True`` if the Tensor is quantized, ``False`` otherwise.
 |  
 |  iiss__ssppaarrssee
 |      Is ``True`` if the Tensor uses sparse storage layout, ``False`` otherwise.
 |  
 |  iiss__vvuullkkaann
 |  
 |  iiss__xxppuu
 |      Is ``True`` if the Tensor is stored on the XPU, ``False`` otherwise.
 |  
 |  llaayyoouutt
 |  
 |  nnaammee
 |  
 |  nnaammeess
 |      Stores names for each of this tensor's dimensions.
 |      
 |      ``names[idx]`` corresponds to the name of tensor dimension ``idx``.
 |      Names are either a string if the dimension is named or ``None`` if the
 |      dimension is unnamed.
 |      
 |      Dimension names may contain characters or underscore. Furthermore, a dimension
 |      name must be a valid Python variable name (i.e., does not start with underscore).
 |      
 |      Tensors may not have two named dimensions with the same name.
 |      
 |      .. warning::
 |          The named tensor API is experimental and subject to change.
 |  
 |  nnddiimm
 |      Alias for :meth:`~Tensor.dim()`
 |  
 |  oouuttppuutt__nnrr
 |  
 |  rreeaall
 |      Returns a new tensor containing real values of the :attr:`self` tensor.
 |      The returned tensor and :attr:`self` share the same underlying storage.
 |      
 |      .. warning::
 |          :func:`real` is only supported for tensors with complex dtypes.
 |      
 |      Example::
 |          >>> x=torch.randn(4, dtype=torch.cfloat)
 |          >>> x
 |          tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
 |          >>> x.real
 |          tensor([ 0.3100, -0.5445, -1.6492, -0.0638])
 |  
 |  rreeqquuiirreess__ggrraadd
 |      Is ``True`` if gradients need to be computed for this Tensor, ``False`` otherwise.
 |      
 |      .. note::
 |      
 |          The fact that gradients need to be computed for a Tensor do not mean that the :attr:`grad`
 |          attribute will be populated, see :attr:`is_leaf` for more details.
 |  
 |  sshhaappee
 |  
 |  vvoollaattiillee
