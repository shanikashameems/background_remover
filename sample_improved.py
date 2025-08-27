import streamlit as st
from rembg import remove
from PIL import Image
import numpy as np
import cv2
import io

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except Exception:
    CRF_AVAILABLE = False

st.set_page_config(page_title="Premium Background Remover", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #03fcfc 0%, #e6ffff 100%);
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            max-width: 950px;
            margin: auto;
            padding: 2rem;
            background: white;
            border-radius: 20px;
            box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #03fcfc;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #03fcfc;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #00e0e0;
        }
        .css-1v0mbdj, .css-1kyxreq {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🖼️ Premium Background Remover</h1>", unsafe_allow_html=True)

def pil_to_bgr(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bytes_to_pngbytes(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def rembg_alpha(image_bytes):
    out = remove(image_bytes)
    pil = Image.open(io.BytesIO(out)).convert("RGBA")
    alpha = np.array(pil)[:, :, 3].astype(np.uint8)
    return alpha, pil

def refine_with_grabcut(bgr_img, alpha_uint8, iter_count=5):
    h, w = alpha_uint8.shape
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[alpha_uint8 >= 240] = cv2.GC_FGD
    mask[alpha_uint8 <= 10] = cv2.GC_BGD
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr_img.copy(), mask, None, bgModel, fgModel, iter_count, mode=cv2.GC_INIT_WITH_MASK)
    except:
        return (alpha_uint8 > 127).astype(np.uint8) * 255
    result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8) * 255
    return result_mask

def apply_dense_crf(bgr_img, mask_prob):
    if not CRF_AVAILABLE:
        return mask_prob
    h, w = mask_prob.shape
    softmax = np.stack([1.0 - mask_prob, mask_prob], axis=0)
    unary = unary_from_softmax(softmax)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=bgr_img, compat=10)
    Q = d.inference(5)
    return np.array(Q).reshape((2, h, w))[1]

def smooth_alpha(alpha_float, kernel):
    if kernel <= 1:
        return alpha_float
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(alpha_float.astype(np.float32), (k, k), 0)

def compose_rgba(bgr_img, alpha_float):
    alpha_uint8 = (np.clip(alpha_float, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha_uint8])
    return Image.fromarray(rgba)

uploaded_file = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
st.markdown("### 🎛 Refinement Controls")
cols = st.columns(3)
grabcut_iters = cols[0].slider("GrabCut Iterations", 1, 10, 5)
smooth_kernel = cols[1].slider("Edge Smoothing", 1, 21, 7, step=2)
use_crf = cols[2].checkbox("Use DenseCRF", value=False)
if use_crf and not CRF_AVAILABLE:
    st.warning("DenseCRF not installed. Install `pydensecrf` to enable this feature.")

st.markdown("---")

if uploaded_file is None:
    st.info("Upload an image to start background removal.")
else:
    input_pil = Image.open(uploaded_file).convert("RGB")
    bgr = pil_to_bgr(input_pil)
    buf = io.BytesIO()
    input_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with st.spinner("Removing background and refining..."):
        try:
            initial_alpha, _ = rembg_alpha(img_bytes)
        except Exception as e:
            st.error(f"Error in rembg: {e}")
            st.stop()

        grabcut_mask = refine_with_grabcut(bgr, initial_alpha, iter_count=grabcut_iters)
        prob = grabcut_mask.astype(np.float32) / 255.0

        if use_crf and CRF_AVAILABLE:
            prob = apply_dense_crf(bgr, prob)

        prob_smoothed = smooth_alpha(prob, smooth_kernel)
        prob_smoothed = (prob_smoothed - prob_smoothed.min()) / (prob_smoothed.max() - prob_smoothed.min() + 1e-8)
        final_pil = compose_rgba(bgr, prob_smoothed)

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(input_pil, use_container_width=True)
    col2.subheader("Final Output")
    col2.image(final_pil, use_container_width=True)

    # Download button
    final_bytes = bytes_to_pngbytes(final_pil)
    st.download_button("📥 Download Final Image", data=final_bytes, file_name="output_refined.png", mime="image/png")

