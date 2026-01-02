"""GlobalConcat and GlobalStack connectors for merging parallel branch outputs."""

from typing import Tuple
from pydantic import Field
import torch

from vectormesh.types import VectorMeshComponent, TwoDTensor, ThreeDTensor, FourDTensor, NDTensor, VectorMeshError


class GlobalConcat(VectorMeshComponent):
    """Concatenate parallel branch outputs along specified dimension.

    Merges multiple tensor outputs from Parallel combinator into a single
    tensor by concatenating along the specified dimension (typically feature
    dimension). Requires all inputs to have same dimensionality (all 2D or
    all 3D).

    Args:
        dim: Dimension along which to concatenate (required, use dim=1)

    Example:
        ```python
        parallel = Parallel([TwoDVectorizer("model1"), TwoDVectorizer("model2")])
        concat = GlobalConcat(dim=1)
        pipeline = parallel >> concat
        result = pipeline(["Hello"])  # TwoDTensor[batch, dim1+dim2]
        ```

    Shapes:
        Input: Tuple[TwoDTensor[B,D1], TwoDTensor[B,D2], ...] (same dimensionality)
        Output: TwoDTensor[B, D1+D2+...] (concatenated features)

    Behavior Matrix:
        | Input 1      | Input 2      | Output            | Notes                           |
        |--------------|--------------|-------------------|---------------------------------|
        | TwoDTensor   | TwoDTensor   | TwoDTensor        | Concat features (B,D1+D2)       |
        | ThreeDTensor | ThreeDTensor | ThreeDTensor      | Concat chunks (B,C1+C2,E)       |
        | TwoDTensor   | ThreeDTensor | ERROR             | Educational error - use Stack   |
    """

    dim: int = Field(description="Dimension along which to concatenate")

    @classmethod
    def infer_output_type(cls, input_tuple_types: Tuple[type, ...]) -> type:
        """Infer output tensor type from input tuple types.

        For definition-time validation in Serial chains. Returns the tensor
        type that will be output given the input tuple types.

        Args:
            input_tuple_types: Tuple of tensor types (e.g., (TwoDTensor, TwoDTensor))

        Returns:
            Output tensor type (same dimensionality as inputs)

        Raises:
            VectorMeshError: If mixed dimensionality (2D+3D) detected
        """
        if not input_tuple_types:
            raise VectorMeshError(
                message="GlobalConcat requires at least one input tensor",
                hint="Provide tuple of tensors to concatenate",
                fix="Ensure Parallel combinator has at least one branch"
            )

        # Check if all inputs are 2D
        all_2d = all(t == TwoDTensor for t in input_tuple_types)
        # Check if all inputs are 3D
        all_3d = all(t == ThreeDTensor for t in input_tuple_types)

        if all_2d:
            return TwoDTensor
        elif all_3d:
            return ThreeDTensor
        else:
            # Mixed dimensionality
            raise VectorMeshError(
                message="GlobalConcat cannot merge mixed 2D/3D tensors from Parallel branches",
                hint="Either normalize dimensions or use stacking",
                fix="Option 1: Add MeanAggregator() to 3D branch to normalize dimensions. "
                    "Option 2: Use GlobalStack(dim=1) instead, which handles mixed dimensionality by creating chunk dimension"
            )

    def __call__(self, inputs: Tuple[NDTensor, ...]) -> NDTensor:
        """Concatenate tensors along specified dimension.

        Validates dimensionality compatibility and batch dimension matching,
        then concatenates all tensors along the specified dimension.

        Args:
            inputs: Tuple of tensors from Parallel combinator output

        Returns:
            Single concatenated tensor

        Raises:
            VectorMeshError: If dimensionality mismatch or batch dimensions don't match

        Shapes:
            Input: Tuple of same-dimensionality tensors
            Output: Single tensor with concatenated dimension
        """
        if not inputs:
            raise VectorMeshError(
                message="GlobalConcat received empty input tuple",
                hint="Provide at least one tensor to concatenate",
                fix="Check Parallel combinator configuration"
            )

        # Check dimensionality consistency
        dims = [t.ndim for t in inputs]
        if len(set(dims)) > 1:
            # Mixed dimensionality
            raise VectorMeshError(
                message=f"GlobalConcat cannot merge mixed dimensionality tensors (found {set(dims)}D)",
                hint="All inputs must have same dimensionality (all 2D or all 3D)",
                fix="Option 1: Add MeanAggregator() to normalize 3D→2D. "
                    "Option 2: Use GlobalStack(dim=1) which handles mixed dimensions"
            )

        # Validate batch dimensions match
        batch_sizes = [t.shape[0] for t in inputs]
        if len(set(batch_sizes)) > 1:
            raise VectorMeshError(
                message=f"GlobalConcat requires matching batch dimensions (found {batch_sizes})",
                hint="All tensors must have same batch size in dimension 0",
                fix="Ensure all Parallel branches process same batch of inputs"
            )

        # Perform concatenation
        try:
            result = torch.cat(inputs, dim=self.dim)
            return result
        except RuntimeError as e:
            raise VectorMeshError(
                message=f"Concatenation failed: {str(e)}",
                hint="Check tensor shapes and dimension parameter",
                fix=f"Verify dim={self.dim} is valid for input tensors with shape {inputs[0].shape}"
            ) from e


class GlobalStack(VectorMeshComponent):
    """Stack parallel branch outputs along new or existing dimension.

    Merges multiple tensor outputs by stacking along a dimension, either
    creating a new chunk dimension (2D→3D) or extending an existing chunk
    dimension (3D+2D→3D). Handles padding for mismatched dimensions.

    Args:
        dim: Dimension along which to stack (default: 1 for chunks)

    Example:
        ```python
        # Create new chunk dimension
        parallel = Parallel([TwoDVectorizer("model1"), TwoDVectorizer("model2")])
        stack = GlobalStack(dim=1)
        result = stack(parallel(["Hello"]))  # ThreeDTensor[batch, 2, emb]

        # Extend chunk dimension
        parallel_mixed = Parallel([ThreeDVectorizer("model"), TwoDVectorizer("model2")])
        stack = GlobalStack(dim=1)
        result = stack(parallel_mixed(["Hello"]))  # ThreeDTensor[batch, chunks+1, emb]
        ```

    Shapes:
        Input 2D+2D: Tuple[TwoDTensor[B,E1], TwoDTensor[B,E2]] → ThreeDTensor[B,2,max(E1,E2)]
        Input 3D+2D: Tuple[ThreeDTensor[B,C,E], TwoDTensor[B,E]] → ThreeDTensor[B,C+1,E]
        Input 3D+3D: Tuple[ThreeDTensor[B,C1,E], ThreeDTensor[B,C2,E]] → FourDTensor[B,2,max(C1,C2),E]

    Behavior Matrix:
        | Input 1      | Input 2      | Output            | Notes                                 |
        |--------------|--------------|-------------------|---------------------------------------|
        | TwoDTensor   | TwoDTensor   | ThreeDTensor      | Create chunks (B,2,max(E1,E2))        |
        | ThreeDTensor | TwoDTensor   | ThreeDTensor      | Extend chunks (B,C+1,E)               |
        | ThreeDTensor | ThreeDTensor | FourDTensor       | Multi-branch (B,2,max(C1,C2),E)       |
    """

    dim: int = Field(default=1, description="Dimension along which to stack")

    @classmethod
    def infer_output_type(cls, input_tuple_types: Tuple[type, ...]) -> type:
        """Infer output tensor type from input tuple types.

        For definition-time validation in Serial chains. GlobalStack returns
        ThreeDTensor for 2D inputs or mixed 2D/3D, FourDTensor for 3D+3D.

        Args:
            input_tuple_types: Tuple of tensor types (e.g., (TwoDTensor, TwoDTensor))

        Returns:
            ThreeDTensor for 2D+2D or 3D+2D cases (creates/extends chunks)
            FourDTensor for 3D+3D case (creates branch dimension)

        Raises:
            VectorMeshError: If embedding dimensions incompatible
        """
        if not input_tuple_types:
            raise VectorMeshError(
                message="GlobalStack requires at least one input tensor",
                hint="Provide tuple of tensors to stack",
                fix="Ensure Parallel combinator has at least one branch"
            )

        # Check if all inputs are 3D
        all_3d = all(t == ThreeDTensor for t in input_tuple_types)
        if all_3d:
            return FourDTensor  # 3D+3D creates 4D multi-branch
        else:
            return ThreeDTensor  # 2D+2D or mixed 2D/3D creates/extends 3D

    def __call__(self, inputs: Tuple[NDTensor, ...]) -> NDTensor:
        """Stack tensors along specified dimension, creating or extending chunk dimension.

        Handles padding for mismatched dimensions and proper dimension
        management for all cases: 2D→3D, 3D extension, and 3D→4D multi-branch.

        Args:
            inputs: Tuple of tensors from Parallel combinator output

        Returns:
            ThreeDTensor for 2D+2D or 3D+2D cases
            FourDTensor for 3D+3D case (multi-branch representation)

        Raises:
            VectorMeshError: If shapes incompatible or dimensions don't align

        Shapes:
            2D+2D: [B,E1], [B,E2] → ThreeDTensor[B,2,max(E1,E2)] (pad embeddings)
            3D+2D: [B,C,E], [B,E] → ThreeDTensor[B,C+1,E] (unsqueeze 2D first)
            3D+3D: [B,C1,E], [B,C2,E] → FourDTensor[B,2,max(C1,C2),E] (pad chunks, create branch dim)
        """
        if not inputs:
            raise VectorMeshError(
                message="GlobalStack received empty input tuple",
                hint="Provide at least one tensor to stack",
                fix="Check Parallel combinator configuration"
            )

        # Validate batch dimensions match
        batch_sizes = [t.shape[0] for t in inputs]
        if len(set(batch_sizes)) > 1:
            raise VectorMeshError(
                message=f"GlobalStack requires matching batch dimensions (found {batch_sizes})",
                hint="All tensors must have same batch size in dimension 0",
                fix="Ensure all Parallel branches process same batch of inputs"
            )

        # Determine dimensionality combination
        dims = [t.ndim for t in inputs]
        all_2d = all(d == 2 for d in dims)
        all_3d = all(d == 3 for d in dims)
        has_2d = any(d == 2 for d in dims)
        has_3d = any(d == 3 for d in dims)

        batch_size = inputs[0].shape[0]

        if all_2d:
            # 2D+2D: Create chunk dimension, pad embeddings to max
            return self._stack_2d_inputs(inputs, batch_size)
        elif has_2d and has_3d:
            # Mixed 3D+2D: Extend chunk dimension
            return self._stack_mixed_inputs(inputs, batch_size)
        elif all_3d:
            # 3D+3D: Create branch dimension (4D output)
            return self._stack_3d_inputs(inputs, batch_size)
        else:
            raise VectorMeshError(
                message=f"GlobalStack received unexpected tensor dimensions: {dims}",
                hint="Only 2D and 3D tensors are supported",
                fix="Check input tensor shapes and dimensions"
            )

    def _stack_2d_inputs(self, inputs: Tuple[NDTensor, ...], batch_size: int) -> ThreeDTensor:
        """Stack 2D tensors into 3D by creating chunk dimension.

        Args:
            inputs: Tuple of 2D tensors [B, E1], [B, E2], ...
            batch_size: Batch dimension size

        Returns:
            ThreeDTensor[B, num_inputs, max_embed]
        """
        # Find max embedding dimension
        embed_dims = [t.shape[1] for t in inputs]
        max_embed = max(embed_dims)

        # Pad all tensors to max embedding dimension
        padded_tensors = []
        for tensor in inputs:
            if tensor.shape[1] < max_embed:
                # Pad on the right side of embedding dimension
                pad_size = max_embed - tensor.shape[1]
                padded = torch.nn.functional.pad(tensor, (0, pad_size))
                padded_tensors.append(padded)
            else:
                padded_tensors.append(tensor)

        # Unsqueeze to add chunk dimension and stack
        unsqueezed = [t.unsqueeze(1) for t in padded_tensors]  # [B, 1, E]
        result = torch.cat(unsqueezed, dim=1)  # [B, num_inputs, max_E]
        return result

    def _stack_mixed_inputs(self, inputs: Tuple[NDTensor, ...], batch_size: int) -> ThreeDTensor:
        """Stack mixed 3D+2D tensors by extending chunk dimension.

        Args:
            inputs: Tuple of 3D and 2D tensors
            batch_size: Batch dimension size

        Returns:
            ThreeDTensor[B, total_chunks, E]
        """
        # Validate embedding dimensions match
        embed_dims = []
        for tensor in inputs:
            if tensor.ndim == 2:
                embed_dims.append(tensor.shape[1])
            else:  # 3D
                embed_dims.append(tensor.shape[2])

        if len(set(embed_dims)) > 1:
            raise VectorMeshError(
                message=f"GlobalStack requires matching embedding dimensions (found {embed_dims})",
                hint="All tensors must have same embedding dimension",
                fix="Ensure all Parallel branches output compatible embedding sizes"
            )

        # Unsqueeze 2D tensors to make them 3D [B, 1, E]
        normalized_tensors = []
        for tensor in inputs:
            if tensor.ndim == 2:
                normalized_tensors.append(tensor.unsqueeze(1))
            else:
                normalized_tensors.append(tensor)

        # Concatenate along chunk dimension
        result = torch.cat(normalized_tensors, dim=1)
        return result

    def _stack_3d_inputs(self, inputs: Tuple[NDTensor, ...], batch_size: int) -> FourDTensor:
        """Stack 3D tensors into 4D by creating branch dimension.

        Args:
            inputs: Tuple of 3D tensors [B, C1, E], [B, C2, E], ...
            batch_size: Batch dimension size

        Returns:
            FourDTensor[B, num_inputs, max_chunks, E]
        """
        # Validate embedding dimensions match
        embed_dims = [t.shape[2] for t in inputs]
        if len(set(embed_dims)) > 1:
            raise VectorMeshError(
                message=f"GlobalStack requires matching embedding dimensions (found {embed_dims})",
                hint="All 3D tensors must have same embedding dimension",
                fix="Ensure all Parallel branches output compatible embedding sizes"
            )

        # Find max chunk dimension
        chunk_dims = [t.shape[1] for t in inputs]
        max_chunks = max(chunk_dims)
        embed_dim = inputs[0].shape[2]

        # Pad all tensors to max chunk dimension
        padded_tensors = []
        for tensor in inputs:
            if tensor.shape[1] < max_chunks:
                # Pad on chunks dimension (dim=1)
                pad_size = max_chunks - tensor.shape[1]
                # Pad format: (left, right) for last dim, then (left, right) for second-to-last, etc.
                # We want to pad dim=1, so pad format is (0, 0, 0, pad_size, 0, 0) for (E_left, E_right, C_left, C_right, B_left, B_right)
                # Actually simpler: pad along dim=1
                padding = torch.zeros(batch_size, pad_size, embed_dim, device=tensor.device, dtype=tensor.dtype)
                padded = torch.cat([tensor, padding], dim=1)
                padded_tensors.append(padded)
            else:
                padded_tensors.append(tensor)

        # Stack along new branch dimension (dim=1)
        result = torch.stack(padded_tensors, dim=1)  # [B, num_inputs, max_chunks, E]
        return result
