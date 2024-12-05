import torch


def load_tensor(file_path):
    """Load a tensor from a .pth file."""
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"Error loading tensor from {file_path}: {e}")
        return None


def compare_tensors(tensor1, tensor2):
    """Compare two tensors for equality."""
    if tensor1.shape != tensor2.shape:
        print("Tensors have different shapes.")
        return False

    if not torch.equal(tensor1, tensor2):
        print("Tensors have different values.")
        return False

    return True


def main():
    # Load tensors
    t1 = load_tensor("t1.pth")
    t2 = load_tensor("t2.pth")

    if t1 is None or t2 is None:
        print("One or both tensors could not be loaded.")
        return

    # Compare tensors
    if compare_tensors(t1, t2):
        print("Tensors have the exact same values.")
    else:
        print("Tensors do not have the exact same values.")


if __name__ == "__main__":
    main()
