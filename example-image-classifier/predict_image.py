import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

from image_model import SimpleCNN, ResNetCIFAR

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path='image_classifier.pth', model_type='simple', device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        model_type: Type of model ('simple' or 'resnet')
        device: Device to load the model on
        
    Returns:
        Loaded model in evaluation mode
    """
    # Initialize model architecture
    if model_type == 'simple':
        model = SimpleCNN(num_classes=10)
    else:
        model = ResNetCIFAR(num_classes=10)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for CIFAR-10 model inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 uses 32x32 images
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict_image(model, image_tensor, device='cpu', top_k=3):
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples
    """
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Forward pass (no gradient computation needed)
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Convert to Python lists
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
    
    # Create list of (class_name, probability) tuples
    predictions = [
        (CLASSES[idx], prob * 100)
        for idx, prob in zip(top_indices, top_probs)
    ]
    
    return predictions


def predict_from_file(image_path, model_path='image_classifier.pth', 
                     model_type='simple', device='cpu'):
    """
    Complete pipeline: load model and predict on an image file.
    
    Args:
        image_path: Path to the image to classify
        model_path: Path to the saved model checkpoint
        model_type: Type of model ('simple' or 'resnet')
        device: Device to run inference on
    """
    print("=" * 60)
    print("CIFAR-10 Image Classifier - Inference")
    print("=" * 60)
    print()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using 'python train_image.py'")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, model_type, device)
    print()
    
    # Preprocess image
    print(f"Loading image: {image_path}")
    image_tensor = preprocess_image(image_path)
    print(f"Image shape: {image_tensor.shape}")
    print()
    
    # Make prediction
    print("Making prediction...")
    predictions = predict_image(model, image_tensor, device, top_k=5)
    print()
    
    # Display results
    print("=" * 60)
    print("Predictions (Top 5)")
    print("=" * 60)
    for i, (class_name, prob) in enumerate(predictions, 1):
        bar = '█' * int(prob / 2)  # Simple progress bar
        print(f"{i}. {class_name:12s} {prob:5.2f}% {bar}")
    print()
    
    # Show top prediction
    top_class, top_prob = predictions[0]
    print(f"✓ Predicted class: {top_class.upper()} ({top_prob:.2f}% confidence)")
    print()


def predict_from_dataset(model_path='image_classifier.pth', model_type='simple', 
                        num_samples=10, device='cpu'):
    """
    Make predictions on random samples from the CIFAR-10 test set.
    
    Args:
        model_path: Path to the saved model checkpoint
        model_type: Type of model ('simple' or 'resnet')
        num_samples: Number of random samples to predict
        device: Device to run inference on
    """
    from image_dataset import get_cifar10_loaders
    import random
    
    print("=" * 60)
    print("Testing on CIFAR-10 Test Set")
    print("=" * 60)
    print()
    
    # Load model
    model = load_model(model_path, model_type, device)
    print()
    
    # Load test dataset
    _, _, test_loader, _ = get_cifar10_loaders(batch_size=1)
    
    # Get all test samples
    test_samples = list(test_loader)
    
    # Randomly select samples
    selected_samples = random.sample(test_samples, min(num_samples, len(test_samples)))
    
    correct = 0
    total = 0
    
    print(f"Predicting on {len(selected_samples)} random samples...")
    print("=" * 60)
    
    for i, (image, label) in enumerate(selected_samples, 1):
        # Make prediction
        predictions = predict_image(model, image, device, top_k=1)
        predicted_class, confidence = predictions[0]
        true_class = CLASSES[label.item()]
        
        # Check if correct
        is_correct = predicted_class == true_class
        correct += is_correct
        total += 1
        
        # Display result
        status = "✓" if is_correct else "✗"
        print(f"{i:2d}. {status} True: {true_class:12s} | "
              f"Predicted: {predicted_class:12s} ({confidence:.1f}%)")
    
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.2f}%)")
    print()


if __name__ == "__main__":
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # User provided an image path
        image_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'simple'
        predict_from_file(image_path, model_type=model_type, device=device)
    else:
        # No arguments: test on random samples from dataset
        print("Usage: python predict_image.py <image_path> [model_type]")
        print("  model_type: 'simple' (default) or 'resnet'")
        print()
        print("No image provided. Testing on random samples from CIFAR-10 test set...")
        print()
        
        predict_from_dataset(model_type='simple', num_samples=10, device=device)
