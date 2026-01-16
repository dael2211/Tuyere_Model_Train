import torch



def export_to_onnx(model_path, output_onnx_path, input_shape=(1, 3, 688, 800), dynamic_batch=False):
    """
    Export a PyTorch segmentation model to ONNX format with adjusted input shape.

    Args:
        model_path (str): Path to the trained PyTorch model (.pt).
        output_onnx_path (str): Path to save the ONNX model.
        input_shape (tuple): Input shape as (batch_size, channels, height, width).
        dynamic_batch (bool): Whether to allow dynamic batch sizes in the ONNX model.
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)  # Load the model
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(*input_shape, device=device)

    # ONNX export parameters
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # export to ONNX
    torch.onnx.export(
        model,
        dummy_input,  # Dummy input
        output_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,  #
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes  #dynamic batching
    )
    print(f"Model exported to ONNX format at: {output_onnx_path}")


model_path = r"C:\Users\DamianGentner\Documents\tuyere_camera\trainig_results\exp26\models\combustion_model.pt"
output_onnx_path = r"C:\Users\DamianGentner\Documents\tuyere_camera\trainig_results\exp26\models\combustion_model.onnx"


export_to_onnx(model_path, output_onnx_path, input_shape=(1, 3, 688, 800), dynamic_batch=True)
