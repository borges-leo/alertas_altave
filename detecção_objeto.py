import cv2
import torch
import json
import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Objetos - Interface Gráfica")
        self.image_path = None
        self.model = self.initialize_model()
        self.annotations = []

        self.label = Label(root, text="Selecione uma imagem para começar")
        self.label.pack()

        self.canvas = Label(root)
        self.canvas.pack()

        self.load_button = Button(root, text="Carregar Imagem", command=self.load_image)
        self.load_button.pack()

        self.validate_button = Button(root, text="Validar Detecção", command=self.save_annotations, state="disabled")
        self.validate_button.pack()

        self.incorrect_button = Button(root, text="Marcar como Incorreto", command=self.mark_incorrect, state="disabled")
        self.incorrect_button.pack()

        self.next_button = Button(root, text="Próxima Imagem", command=self.clear, state="disabled")
        self.next_button.pack()

    def initialize_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        model.eval()
        return model

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if not self.image_path:
            return

        self.process_image()
        self.validate_button["state"] = "normal"
        self.incorrect_button["state"] = "normal"
        self.next_button["state"] = "normal"

    def process_image(self):
        image = cv2.imread(self.image_path)
        original_image = image.copy()

        transform = F.to_tensor
        tensor_image = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(tensor_image)

        self.detections = []
        for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                self.detections.append({"box": (x1, y1, x2, y2), "score": float(score)})

                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_image, f"{score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.config(image=self.tk_image)

    def save_annotations(self):
        feedback = messagebox.askyesno("Validar", "As caixas estão corretas?")
        annotation = {
            "image": os.path.basename(self.image_path),
            "detections": self.detections,
            "validated": feedback
        }
        self.annotations.append(annotation)

        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

        messagebox.showinfo("Salvo", "Anotação salva com sucesso!")

    def mark_incorrect(self):
        """
        Marca a imagem como completamente incorreta.
        """
        with open("incorrect_images.json", "a") as f:
            json.dump({"image": os.path.basename(self.image_path)}, f)
            f.write("\n")

        messagebox.showinfo("Erro Registrado", "A imagem foi marcada como incorreta.")

    def clear(self):
        self.canvas.config(image="")
        self.label.config(text="Selecione uma nova imagem")
        self.image_path = None
        self.validate_button["state"] = "disabled"
        self.incorrect_button["state"] = "disabled"
        self.next_button["state"] = "disabled"

if __name__ == "__main__":
    root = Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
