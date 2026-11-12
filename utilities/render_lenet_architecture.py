"""Render the implemented simplified LeNet card in the existing Pillow style."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from matplotlib.font_manager import FontProperties, findfont
import torch

from scripts.model.digit_model_definitions import LeNet


def forward_shapes():
    """Inspect a synthetic CPU input, without training or loading any dataset."""
    model = LeNet().eval()
    features = torch.zeros(1, 1, 28, 28)
    shapes = []
    with torch.no_grad():
        for layer in model.network:
            features = layer(features)
            shapes.append(tuple(features.shape[1:]))
    expected = [(6,28,28),(6,28,28),(6,14,14),
                (16,10,10),(16,10,10),(16,5,5),
                (400,),(120,),(120,),(84,),(84,),(10,)]
    if shapes != expected:
        raise ValueError(f'LeNet changed; update the architecture card: {shapes}')
    return shapes


def render_card(output):
    shapes = forward_shapes()
    width, height = 1800, 1310
    image = Image.new('RGB', (width, height), '#f6f7f9')
    draw = ImageDraw.Draw(image)
    regular = findfont(FontProperties(family='DejaVu Sans'))
    bold = findfont(FontProperties(family='DejaVu Sans', weight='bold'))
    font = lambda size, heavy=False: ImageFont.truetype(bold if heavy else regular, size)
    ink, muted, accent = '#17243a', '#536174', '#245aa8'
    draw.rounded_rectangle((35,30,width-35,height-30),radius=28,fill='white')
    draw.text((80,65),'Simplified LeNet CNN',font=font(65,True),fill=ink)
    draw.text((82,148),'Shapes: height × width × feature maps',font=font(29),fill=muted)
    draw.text((80,218),'Input',font=font(47,True),fill=accent)
    draw.text((80,285),'28 × 28 × 1',font=font(44),fill=ink)
    draw.text((80,374),'Down',font=font(47,True),fill=accent)
    draw.text((80,437),'One convolution per row: stride 1, tanh after each.',font=font(27),fill=muted)
    draw.text((80,480),'Average pool: 2 × 2, stride 2.',font=font(27),fill=muted)
    columns = [80,310,500,770,1230,1720]
    top = 545
    draw.rounded_rectangle((80,top,1720,top+86),radius=10,fill='#eaf0fa')
    headers = ['Filter size','Padding','Feature maps','After convolution','After average pool 2']
    for x,label in zip(columns,headers):
        draw.text((x+20,top+24),label,font=font(28,True),fill=ink)
    for row,(conv_index,pool_index) in enumerate(((0,2),(3,5))):
        channels,h,w = shapes[conv_index]
        pooled,ph,pw = shapes[pool_index]
        y = top+86+row*92
        if row%2==0:
            draw.rectangle((80,y,1720,y+92),fill='#f6f8fb')
        values = ['5 × 5',str((2,0)[row]),str(channels),f'{h} × {w} × {channels}',f'{ph} × {pw} × {pooled}']
        for x,value in zip(columns,values):
            draw.text((x+20,y+27),value,font=font(31),fill=ink)
    draw.line((80,815,1720,815),fill='#d7dfea',width=2)
    draw.text((80,881),'Classifier',font=font(47,True),fill=accent)
    draw.text((80,952),'5 × 5 × 16 → Flatten 400 → Linear 120 + tanh → Linear 84 + tanh',font=font(35),fill=ink)
    draw.text((80,1060),'Output layer',font=font(47,True),fill=accent)
    draw.text((80,1130),'84 → Linear 10 logits',font=font(38),fill=ink)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('Description','Implemented Notebook02 LeNet: 5x5 conv1 padding2, tanh, average pool2; 5x5 conv2 padding0, tanh, average pool2; flatten400, linear120 tanh, linear84 tanh, linear10 raw logits. Shapes checked with a synthetic CPU forward pass.')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output,pnginfo=metadata)
    return image


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    path = root/'docs/evaluations/assets/lenet-architecture.png'
    render_card(path)
    print(path)
