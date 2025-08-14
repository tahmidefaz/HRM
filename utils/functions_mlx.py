import os
import importlib.util


def load_model_class_mlx(model_name: str):
    """Load model class for MLX version"""
    
    # Map original model names to MLX versions
    model_mapping = {
        "models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1": "models.hrm.hrm_act_v1_mlx@HierarchicalReasoningModel_ACTV1",
        "models.losses@ACTLossHead": "models.losses_mlx@ACTLossHead",
    }
    
    # Use mapping if available, otherwise try to append _mlx
    if model_name in model_mapping:
        model_name = model_mapping[model_name]
    elif "@" in model_name:
        module_path, class_name = model_name.split("@")
        if not module_path.endswith("_mlx"):
            module_path += "_mlx"
        model_name = f"{module_path}@{class_name}"
    
    # Parse module and class
    if "@" in model_name:
        module_path, class_name = model_name.split("@")
    else:
        raise ValueError(f"Model name must be in format 'module@class', got: {model_name}")
    
    # Import module
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load model class {class_name} from {module_path}: {e}")


def get_model_source_path_mlx(model_name: str):
    """Get source path for MLX model"""
    
    # Map to MLX version
    model_mapping = {
        "models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1": "models/hrm/hrm_act_v1_mlx.py",
        "models.losses@ACTLossHead": "models/losses_mlx.py",
    }
    
    if model_name in model_mapping:
        return model_mapping[model_name]
    
    # Try to infer MLX path
    if "@" in model_name:
        module_path, _ = model_name.split("@")
        file_path = module_path.replace(".", "/") + "_mlx.py"
        if os.path.exists(file_path):
            return file_path
    
    return None