import modal
from src.modal_runner import inference_image

app = modal.App("cot-adversarial-harness-build-test")

@app.function(image=inference_image)
def dummy():
    pass

if __name__ == "__main__":
    with app.run():
        dummy.remote()
