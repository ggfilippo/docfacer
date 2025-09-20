from ultralytics import YOLO 

def main():
    model = YOLO("yolov10s.pt")

    # Treinar o modelo
    model.train(
        data="C:\\Projetos IFSP\\train-yolo-faceLiveness\\dataset\\dataOffline.yaml", 
        epochs=150,               
        batch=16,                 
        device=0,
    )

    metrics = model.val(device=0) 
    print(metrics)

if __name__ == '__main__':
    main()