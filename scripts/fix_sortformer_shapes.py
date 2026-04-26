"""
Fix dynamic shapes in Sortformer ONNX to static values for a given chunk_len/fifo_len/spkcache_len.
Static shapes allow CUDA EP to fuse MatMuls properly, enabling real GPU acceleration.

Usage:
  python fix_sortformer_shapes.py \
    diar_streaming_sortformer_4spk-v2.onnx \
    diar_sortformer_static62.onnx \
    --chunk-len 62 --fifo-len 62 --spkcache-len 94

After running, test with:
  python -c "
  import onnxruntime as ort, numpy as np
  sess = ort.InferenceSession('diar_sortformer_static62.onnx', providers=['CUDAExecutionProvider'])
  print(sess.get_providers())
  "
"""
import argparse
import onnx
from onnx import TensorProto, helper, shape_inference

SUBSAMPLING = 8
EMB_DIM = 512


def fix_shapes(input_path, output_path, chunk_len, fifo_len, spkcache_len, right_context=1):
    model = onnx.load(input_path)

    chunk_frames_in = chunk_len * SUBSAMPLING      # mel frames fed as input
    chunk_frames_pre = chunk_len + right_context   # after subsampling + right ctx

    # Map symbolic dim names -> concrete values for the primary inputs
    shape_map = {
        "time_chunk": chunk_frames_in,
        "time_cache": spkcache_len,
        "time_fifo":  fifo_len,
        "batch": 1,
    }

    def fix_type(type_proto):
        if not type_proto.HasField("tensor_type"):
            return
        shape = type_proto.tensor_type.shape
        if shape is None:
            return
        for dim in shape.dim:
            if dim.dim_param and dim.dim_param in shape_map:
                dim.ClearField("dim_param")
                dim.dim_value = shape_map[dim.dim_param] if False else shape_map.get(dim.dim_param, 0)
                # re-read after clear
        # second pass (clear then set)
        for dim in shape.dim:
            if dim.dim_param in shape_map:
                val = shape_map[dim.dim_param]
                dim.ClearField("dim_param")
                dim.dim_value = val

    # Fix graph inputs
    for inp in model.graph.input:
        fix_type(inp.type)

    # Fix graph outputs
    for out in model.graph.output:
        fix_type(out.type)

    # Fix value_info (intermediate tensors)
    for vi in model.graph.value_info:
        fix_type(vi.type)

    # Run shape inference to propagate static shapes through the graph
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference warning: {e}")

    # Update metadata
    # Remove old metadata entries for these keys and re-add
    existing = {p.key: p.value for p in model.metadata_props}
    model.metadata_props.clear()
    existing.update({
        "chunk_len": str(chunk_len),
        "fifo_len": str(fifo_len),
        "spkcache_len": str(spkcache_len),
        "right_context": str(right_context),
        "static_shapes": "true",
    })
    for k, v in existing.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))

    onnx.save(model, output_path)
    print(f"Saved static-shape model to: {output_path}")

    # Verify
    m = onnx.load(output_path)
    print("\nInput shapes after fix:")
    for inp in m.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--chunk-len", type=int, default=62)
    parser.add_argument("--fifo-len", type=int, default=62)
    parser.add_argument("--spkcache-len", type=int, default=94)
    parser.add_argument("--right-context", type=int, default=1)
    args = parser.parse_args()
    fix_shapes(args.input_path, args.output_path, args.chunk_len, args.fifo_len, args.spkcache_len, args.right_context)
