import os
import importlib.util


def load_model_class(model_name: str):
    """Load model class"""
    
    # Direct model mapping
    model_mapping = {
        # All models are now direct references
        # "models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1": "models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
    }
    
    # Use mapping if available, otherwise use direct path
    if model_name in model_mapping:
        model_name = model_mapping[model_name]
    elif "@" in model_name:
        module_path, class_name = model_name.split("@")
        # No need to append _mlx anymore
        pass
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


def get_model_source_path(model_name: str):
    """Get source path for model"""
    
    # Direct model path mapping
    model_mapping = {
        "models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1": "models/hrm/hrm_act_v1.py",
        "models.losses@ACTLossHead": "models/losses.py",
    }
    
    if model_name in model_mapping:
        return model_mapping[model_name]
    
    # Try to infer path
    if "@" in model_name:
        module_path, _ = model_name.split("@")
        file_path = module_path.replace(".", "/") + ".py"
        if os.path.exists(file_path):
            return file_path
    
    return None