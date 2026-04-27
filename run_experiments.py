from src.evaluate import evaluate

for noise in [0.0, 0.1, 0.3, 0.6]:
    acc, conf = evaluate(noise)
    print(f"Noise: {noise} | Accuracy: {acc:.4f} | Confidence: {conf:.4f}")
