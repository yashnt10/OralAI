"""
OSMF / OPMD 4-tier triage — prediction app (no retraining needed).
Uses the saved checkpoint best_4tier_vit.pth and EXACTLY the training-time preprocessing.

Run on Kaggle / Colab / your laptop:
    pip install timm opencv-python-headless gradio
    python app.py --weights best_4tier_vit.pth                # opens a web page (add --share for a public link)
    python app.py --weights best_4tier_vit.pth --image x.jpg  # command-line prediction
"""
import argparse, cv2, numpy as np, torch, timm
from torchvision import transforms

TIERS = ['T0 Normal', 'T1 Variation from normal', 'T2 OPMD (incl. OSMF)', 'T3 Oral cancer']
ACTIONS = ['Routine recall', 'Short-interval review', 'Refer to oral-medicine specialist', 'URGENT referral']

def clinical_preprocessing(img_bgr):
    """Identical to the training script: specular mask -> Telea inpainting -> Shades-of-Gray (p=6)."""
    h, w = img_bgr.shape[:2]; scale = 1024 / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2] / 255.0, hsv[:, :, 1] / 255.0
    med = cv2.medianBlur(hsv[:, :, 2], 21) / 255.0
    mask = ((v >= 0.92) & (s <= 0.18) & ((v - med) >= 0.12)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    inp = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB); n = rgb / 255.0
    valid = ((n >= 0.02) & (n <= 0.98)).all(axis=2) & (mask == 0)
    if valid.sum() > 100:
        e = np.power(np.mean(np.power(n[valid], 6), axis=0), 1 / 6); e = e / np.linalg.norm(e)
        rgb = np.clip(n / (np.sqrt(3) * e) * 255.0, 0, 255).astype(np.uint8)
    return rgb

TF = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(1024), transforms.Resize((224, 224)),
                         transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

def load_model(weights):
    m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    m.load_state_dict(torch.load(weights, map_location='cpu')); m.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return m.to(dev), dev

@torch.no_grad()
def predict(model, dev, img_rgb):
    rgb = clinical_preprocessing(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    p = torch.softmax(model(TF(rgb)[None].to(dev)), 1)[0].cpu().numpy()
    k = int(p.argmax())
    return k, p, rgb

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='best_4tier_vit.pth'); ap.add_argument('--image'); ap.add_argument('--share', action='store_true')
    a = ap.parse_args()
    model, dev = load_model(a.weights)
    if a.image:
        img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
        k, p, _ = predict(model, dev, img)
        print(f'Prediction: {TIERS[k]}  ->  {ACTIONS[k]}')
        for t, v in zip(TIERS, p): print(f'  {t:28s} {100*v:6.2f} %')
    else:
        import gradio as gr
        def ui(img):
            if img is None: return None, '', None
            k, p, rgb = predict(model, dev, img)
            msg = f'## {TIERS[k]}\n**Action:** {ACTIONS[k]}\n\n*Screening aid only, not a diagnosis.*'
            return {t: float(v) for t, v in zip(TIERS, p)}, msg, rgb
        demo = gr.Interface(fn=ui, inputs=gr.Image(type='numpy', label='Smartphone intraoral photo'),
            outputs=[gr.Label(num_top_classes=4, label='Tier probabilities'), gr.Markdown(), gr.Image(label='After glare removal + colour correction')],
            title='Oral Pre-Cancer (OSMF/OPMD) Smartphone Triage', description='Upload an intraoral photograph. ViT-B/16, 4-tier triage.')
        demo.launch(share=a.share)
