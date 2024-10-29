from transformers import pipeline

# Cargar un pipeline preentrenado para análisis de sentimientos
classifier = pipeline("sentiment-analysis")

# Prueba de ejemplo
new_text = "I'm feeling incredibly happy and satisfied with this!"
result = classifier(new_text)
print(f"The sentiment detected is: {result[0]['label']} with a score of {result[0]['score']}")
