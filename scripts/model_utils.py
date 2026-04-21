import torch


DEFAULT_MODEL_CONFIG = {
    "embedding_dimension": 384,
    "context_length": 512,
    "num_blocks": 6,
    "num_heads": 8,
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_model_config(state_dict):
    token_embedding = state_dict["token_embedding.weight"]
    position_embedding = state_dict["position_embedding.weight"]

    block_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    head_indices = {
        int(key.split(".")[4])
        for key in state_dict
        if key.startswith("blocks.") and ".attention.heads." in key and key.split(".")[4].isdigit()
    }

    return {
        "vocab_size": token_embedding.shape[0],
        "embedding_dimension": token_embedding.shape[1],
        "context_length": position_embedding.shape[0],
        "num_blocks": len(block_indices),
        "num_heads": len(head_indices),
    }


def load_checkpoint_payload(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        config = payload.get("config") or infer_model_config(state_dict)
        return payload, state_dict, config

    config = infer_model_config(payload)
    return {"model_state_dict": payload, "config": config}, payload, config
