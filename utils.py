def format_output(label, confidence, uncertainty):

    return {
        "Prediction": label,
        "Confidence": round(confidence, 4),
        "Uncertainty": round(uncertainty, 4)
    }
