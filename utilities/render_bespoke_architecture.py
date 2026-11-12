"""Draw the implemented 224-input bespoke architecture card with Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from matplotlib.font_manager import FontProperties, findfont


def render_card(output):
    width, height = 1800, 1550
    image = Image.new('RGB', (width, height), '#f6f7f9')
    draw = ImageDraw.Draw(image)
    regular = findfont(FontProperties(family='DejaVu Sans'))
    bold = findfont(FontProperties(family='DejaVu Sans', weight='bold'))
    font = lambda size, heavy=False: ImageFont.truetype(bold if heavy else regular, size)
    ink, muted, accent = '#17243a', '#536174', '#245aa8'
    draw.rounded_rectangle((35, 30, width-35, height-30), radius=28, fill='white')
    draw.text((80,65), 'Bespoke CNN', font=font(65,True), fill=ink)
    draw.text((82,148), 'Shapes: height × width × feature maps', font=font(29), fill=muted)
    draw.text((80,218), 'Up', font=font(47,True), fill=accent)
    draw.text((80,287), '28 × 28 × 1  →  56 × 56 × 1  →  112 × 112 × 1  →  224 × 224 × 1', font=font(36), fill=ink)
    draw.text((80,346), 'One direct bilinear resize to 224; align_corners=False. Doublings shown conceptually.', font=font(26), fill=muted)
    draw.text((80,420), 'Down', font=font(47,True), fill=accent)
    draw.text((80,486), 'Two convolutions per row: stride 1, SAME padding, ReLU or tanh after each.', font=font(27), fill=muted)
    draw.text((80,529), 'Max pool: 2 × 2, stride 2, ceil mode. Interpolation adds no new image information.', font=font(27), fill=muted)
    left, right, top = 80,1720,600
    columns=[80,355,665,1200,1720]
    draw.rounded_rectangle((left,top,right,top+86),radius=10,fill='#eaf0fa')
    headers=['Filter size','Feature maps','After each convolution','After max pool 2']
    for x,label in zip(columns,headers):draw.text((x+20,top+24),label,font=font(28,True),fill=ink)
    spatial=224
    for i,kernel in enumerate(range(10,2,-1)):
        channels=2**(i+1);pooled=(spatial+1)//2;y=top+86+i*71
        if i%2==0:draw.rectangle((left,y,right,y+71),fill='#f6f8fb')
        values=[f'{kernel} × {kernel}',str(channels),f'{spatial} × {spatial} × {channels}',f'{pooled} × {pooled} × {channels}']
        for x,value in zip(columns,values):draw.text((x+20,y+17),value,font=font(31),fill=ink)
        spatial=pooled
    bottom=top+86+8*71
    draw.line((left,bottom,right,bottom),fill='#d7dfea',width=2)
    draw.text((80,1320), 'Output layer', font=font(47,True), fill=accent)
    draw.text((80,1390), '1 × 1 × 256  →  Flatten 256  →  Linear 10 logits',font=font(38),fill=ink)
    draw.text((80,1450), 'Raw logits; no softmax in the model.',font=font(27),fill=muted)
    metadata=PngImagePlugin.PngInfo()
    metadata.add_text('Description','Pillow-rendered implemented eight-stage 224-input BespokeCNN_upsampling; two SAME convolutions and ceil-mode max pooling per row.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output,pnginfo=metadata)
    return image


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    path=root/'docs/evaluations/assets/bespoke-224-architecture.png'
    render_card(path)
    print(path)
