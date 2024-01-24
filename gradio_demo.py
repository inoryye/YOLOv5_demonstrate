import torch
import gradio as gr

model = torch.hub.load("./", "custom", path="runs/train/exp11/weights/best.pt", source="local")

title = "基于改进YOLOv5的火灾实时监测演示项目"

desc = """使用YOLOV5m基础模型\n
            改进：\n
            CIOU>>SIOU\n
            双线性差值上采样>>CARAFE上采样算子
       """

base_conf, base_iou = 0.5, 0.45

def det_image(img, conf_thres, iou_thres):
    model.conf = conf_thres
    model.iou = iou_thres
    return model(img).render()[0]

gr.Interface(inputs=["image", gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)], 
             outputs=["image"], 
             fn=det_image,
             title=title,
             description=desc,
             live=True, 
             examples=[["./example1.jpg", base_conf, base_iou], ["./example2.jpg", 0.5, base_iou]]).launch(share=True)

