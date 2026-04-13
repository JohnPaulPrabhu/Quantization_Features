import onnx
from onnx import helper, checker, shape_inference
from typing import List, Optional, Dict, Set


def load_model(model_path: str) -> onnx.ModelProto:
    model = onnx.load(model_path)
    return model


def save_model(model: onnx.ModelProto, output_path: str):
    checker.check_model(model)
    onnx.save(model, output_path)


def build_name_maps(model: onnx.ModelProto):
    graph = model.graph

    node_map = {}
    producer_map = {}   # tensor_name -> node
    consumer_map = {}   # tensor_name -> list[nodes]
    initializer_map = {init.name: init for init in graph.initializer}
    input_map = {inp.name: inp for inp in graph.input}
    value_info_map = {vi.name: vi for vi in graph.value_info}
    output_map = {out.name: out for out in graph.output}

    for node in graph.node:
        if node.name:
            node_map[node.name] = node
        for out in node.output:
            producer_map[out] = node
        for inp in node.input:
            consumer_map.setdefault(inp, []).append(node)

    return {
        "node_map": node_map,
        "producer_map": producer_map,
        "consumer_map": consumer_map,
        "initializer_map": initializer_map,
        "input_map": input_map,
        "value_info_map": value_info_map,
        "output_map": output_map,
    }


def get_tensor_value_info(model: onnx.ModelProto, tensor_name: str):
    graph = model.graph

    for x in graph.input:
        if x.name == tensor_name:
            return x
    for x in graph.value_info:
        if x.name == tensor_name:
            return x
    for x in graph.output:
        if x.name == tensor_name:
            return x
    return None


def extract_between_nodes(
    input_model_path: str,
    output_model_path: str,
    from_node_name: str,
    to_node_name: str,
):
    """
    Extract submodel from from_node to to_node.
    from_node boundary = outputs of from_node
    to_node boundary   = outputs of to_node

    Note:
    extract_model works on tensor names, not node names.
    """
    model = load_model(input_model_path)
    maps = build_name_maps(model)

    if from_node_name not in maps["node_map"]:
        raise ValueError(f"from_node '{from_node_name}' not found")
    if to_node_name not in maps["node_map"]:
        raise ValueError(f"to_node '{to_node_name}' not found")

    from_node = maps["node_map"][from_node_name]
    to_node = maps["node_map"][to_node_name]

    input_names = [x for x in from_node.output if x]
    output_names = [x for x in to_node.output if x]

    if not input_names:
        raise ValueError(f"from_node '{from_node_name}' has no outputs")
    if not output_names:
        raise ValueError(f"to_node '{to_node_name}' has no outputs")

    # Official ONNX helper extracts submodel using tensor names
    onnx.utils.extract_model(
        input_model_path,
        output_model_path,
        input_names=input_names,
        output_names=output_names,
    )


def collect_required_initializers(
    model: onnx.ModelProto,
    nodes: List[onnx.NodeProto],
) -> List[onnx.TensorProto]:
    init_map = {init.name: init for init in model.graph.initializer}
    required = []
    seen = set()

    for node in nodes:
        for inp in node.input:
            if inp in init_map and inp not in seen:
                required.append(init_map[inp])
                seen.add(inp)

    return required


def make_tensor_value_info_fallback(name: str):
    """
    Minimal fallback if shape/type is unavailable.
    Often ONNX checker prefers proper type info, so shape inference is recommended first.
    """
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)


def extract_single_node(
    input_model_path: str,
    output_model_path: str,
    target_node_name: str,
    infer_shapes: bool = True,
):
    """
    Create a new ONNX model containing exactly one node.
    Keeps:
      - the node
      - any required initializers
      - graph inputs for non-initializer inputs
      - graph outputs for node outputs
    """
    model = load_model(input_model_path)

    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass

    maps = build_name_maps(model)

    if target_node_name not in maps["node_map"]:
        raise ValueError(f"target_node '{target_node_name}' not found")

    target_node = maps["node_map"][target_node_name]
    graph = model.graph

    initializer_map = maps["initializer_map"]

    # Inputs for the new graph:
    # include only non-initializer node inputs as graph inputs
    new_inputs = []
    for inp in target_node.input:
        if not inp:
            continue
        if inp in initializer_map:
            continue

        vi = get_tensor_value_info(model, inp)
        if vi is None:
            vi = make_tensor_value_info_fallback(inp)
        new_inputs.append(vi)

    # Outputs for the new graph:
    new_outputs = []
    for out in target_node.output:
        if not out:
            continue
        vi = get_tensor_value_info(model, out)
        if vi is None:
            vi = make_tensor_value_info_fallback(out)
        new_outputs.append(vi)

    required_initializers = collect_required_initializers(model, [target_node])

    new_graph = helper.make_graph(
        nodes=[target_node],
        name=f"single_node_{target_node_name}",
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=required_initializers,
        value_info=[],
    )

    new_model = helper.make_model(
        new_graph,
        producer_name="custom_single_node_extractor",
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    checker.check_model(new_model)
    onnx.save(new_model, output_model_path)


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    input_model = "model.onnx"

    # 1) Extract range from node A to node B
    extract_between_nodes(
        input_model_path=input_model,
        output_model_path="subgraph_from_A_to_B.onnx",
        from_node_name="Conv_10",
        to_node_name="Relu_15",
    )

    # 2) Extract only one node
    extract_single_node(
        input_model_path=input_model,
        output_model_path="only_one_node.onnx",
        target_node_name="Conv_10",
    )