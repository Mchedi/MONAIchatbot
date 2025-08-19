# ------------------------------
# File: app.py
# MONAI 1.5 Streamlit Inference App (Fixed Orientation Issue)
# ------------------------------

import os
import io
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import glob
import numpy as np
import torch
import nibabel as nib
import streamlit as st
from huggingface_hub import snapshot_download
import gzip
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, ToTensord,
    EnsureTyped, SqueezeDimd
)
from monai.inferers import sliding_window_inference
from monai.bundle import ConfigParser
from monai.networks.nets import UNet

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(page_title="MONAI 1.5 Web App", layout="wide")
st.title("🩺 MONAI 1.5 – Medical Imaging Inference App")
st.caption("Upload an image, choose a model, run inference, and visualize slices.")

# ------------------------------
# HELPERS
# ------------------------------
def save_uploaded_file(up_file, suffix: Optional[str] = None) -> str:
    if up_file is None:
        return ""
    
    # Get the original filename and extension
    original_name = up_file.name
    if suffix is None:
        suffix = Path(original_name).suffix
        # Handle .nii.gz files properly
        if original_name.endswith('.nii.gz'):
            suffix = '.nii.gz'
        elif original_name.endswith('.gz') and not suffix:
            suffix = '.gz'
    
    # Create temp file with proper extension
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            # Reset file pointer to beginning
            up_file.seek(0)
            f.write(up_file.read())
        
        # Verify file was written
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise ValueError("File was not saved properly")
            
        return path
    except Exception as e:
        # Clean up file descriptor if something goes wrong
        try:
            os.close(fd)
        except:
            pass
        raise e

def extract_zip_to_temp(zip_bytes: bytes) -> str:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(temp_dir)
    return temp_dir

def make_default_network(in_channels=1, out_channels=2, is_3d=True):
    dims = 3 if is_3d else 2
    return UNet(
        spatial_dims=dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

def load_bundle_from_local(bundle_dir: str):
    parser = ConfigParser()
    infer_cfg = {"roi_size":[128,128,128],"sw_batch_size":1,"overlap":0.25}
    net = make_default_network()
    
    for cfg_file in ["configs/inference.json", "inference.json"]:
        fpath = Path(bundle_dir) / cfg_file
        if fpath.exists():
            parser.read_config(fpath)
            try:
                net = parser.get_parsed_content("network")
                infer_cfg = parser.get_parsed_content("inferer")
            except Exception:
                pass
            break
    return net, infer_cfg

def load_bundle_from_hf(repo_id: str) -> Tuple[torch.nn.Module, dict, str]:
    cache_dir = snapshot_download(repo_id=repo_id)
    net, infer_cfg = load_bundle_from_local(cache_dir)
    return net, infer_cfg, cache_dir

def build_transforms(is_3d: bool, target_spacing: Tuple[float, ...], intensity_range: Tuple[float, float], is_dicom: bool = False, keys=("image",)):
    tr = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
        EnsureChannelFirstd(keys=keys),
    ]
    
    # Add dimension squeezing for 4D+ volumes to handle extra dimensions
    # This will remove singleton dimensions that might cause orientation issues
    tr.append(SqueezeDimd(keys=keys, dim=None))  # Remove all singleton dimensions
    
    # Re-ensure channel first after squeezing
    tr.append(EnsureChannelFirstd(keys=keys))
    
    # Only apply orientation if not DICOM and we have proper spatial dimensions
    if not is_dicom:
        try:
            tr.append(Orientationd(keys=keys, axcodes="RAS"))
        except Exception:
            # If orientation fails, skip it and continue with other transforms
            st.warning("Skipping orientation transform due to incompatible data dimensions")
    
    tr.extend([
        Spacingd(keys=keys, pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys, 
            a_min=intensity_range[0], 
            a_max=intensity_range[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=keys, source_key=keys[0]),
        EnsureTyped(keys=keys),
        ToTensord(keys=keys),
    ])
    return Compose(tr)

def build_safe_transforms(is_3d: bool, target_spacing: Tuple[float, ...], intensity_range: Tuple[float, float], is_dicom: bool = False, keys=("image",)):
    """Alternative transform pipeline that's more robust to dimension issues"""
    tr = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
        EnsureChannelFirstd(keys=keys),
    ]
    
    # Skip orientation for problematic cases
    if not is_dicom and is_3d:
        tr.append(Orientationd(keys=keys, axcodes="RAS"))
    
    tr.extend([
        Spacingd(keys=keys, pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys, 
            a_min=intensity_range[0], 
            a_max=intensity_range[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=keys, source_key=keys[0]),
        EnsureTyped(keys=keys),
        ToTensord(keys=keys),
    ])
    return Compose(tr)

def visualize_slices(volume: np.ndarray, mask: Optional[np.ndarray] = None, title: str = ""):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    
    # Handle different volume dimensions - more robust approach
    original_shape = volume.shape
    st.info(f"Volume shape before processing: {original_shape}")
    
    # Handle multi-channel volumes (4D: channels, depth, height, width)
    if volume.ndim == 4:
        if volume.shape[0] == 1:
            # Single channel case
            volume = volume.squeeze(0)
            st.info(f"Squeezed single channel volume to shape: {volume.shape}")
        else:
            # Multi-channel case - take first channel or create composite
            st.warning(f"Multi-channel volume detected ({volume.shape[0]} channels). Using first channel for visualization.")
            volume = volume[0]  # Take first channel
            st.info(f"Using first channel, new shape: {volume.shape}")
    
    # Handle mask dimensions similarly
    if mask is not None:
        if mask.ndim == 4:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                mask = mask[0]  # Take first channel
    
    # Now handle 3D vs 2D visualization
    if volume.ndim == 3:
        # 3D volume - show central slices in three views
        c = [s // 2 for s in volume.shape]
        slices = [
            (volume[c[0], :, :], mask[c[0], :, :] if mask is not None else None, "Axial"),
            (volume[:, c[1], :], mask[:, c[1], :] if mask is not None else None, "Coronal"),
            (volume[:, :, c[2]], mask[:, :, c[2]] if mask is not None else None, "Sagittal"),
        ]
    elif volume.ndim == 2:
        # 2D image
        slices = [(volume, mask, "2D")]
    else:
        st.error(f"Cannot visualize volume with {volume.ndim} dimensions")
        return

    # Create visualization
    cols = st.columns(len(slices))
    for col, (img, msk, name) in zip(cols, slices):
        try:
            fig, ax = plt.subplots(figsize=(4,4))
            
            # Ensure image is 2D for matplotlib
            if img.ndim > 2:
                st.warning(f"Image slice still {img.ndim}D, taking first slice/channel")
                img = img.squeeze()
                if img.ndim > 2:
                    img = img[..., 0] if img.shape[-1] < img.shape[0] else img[0]
            
            ax.imshow(img, cmap="gray")
            
            # Handle mask overlay
            if msk is not None:
                if msk.ndim > 2:
                    msk = msk.squeeze()
                    if msk.ndim > 2:
                        msk = msk[..., 0] if msk.shape[-1] < msk.shape[0] else msk[0]
                # Only overlay if mask has non-zero values
                if np.any(msk > 0):
                    ax.imshow(msk, alpha=0.35, cmap="jet")
            
            ax.set_title(f"{title} – {name}")
            ax.axis("off")
            col.pyplot(fig)
            plt.close(fig)
            
        except Exception as viz_error:
            col.error(f"Visualization failed for {name}: {str(viz_error)}")
            st.info(f"Image shape: {img.shape}, Mask shape: {msk.shape if msk is not None else 'None'}")

def validate_and_parse_inputs(spacing_input: str, intensity_input: str, is_3d: bool):
    """Validate and parse spacing and intensity inputs"""
    spacing_values = [x.strip() for x in spacing_input.split(",")]
    expected_dims = 3 if is_3d else 2
    if len(spacing_values) != expected_dims:
        raise ValueError(f"Spacing must have {expected_dims} values for {'3D' if is_3d else '2D'}")
    
    intensity_values = [x.strip() for x in intensity_input.split(",")]
    if len(intensity_values) != 2:
        raise ValueError("Intensity range must have exactly 2 values (a_min,a_max)")
    
    tgt_spacing = tuple(float(x) for x in spacing_values)
    a_min, a_max = (float(x) for x in intensity_values)
    
    return tgt_spacing, (a_min, a_max)

def preprocess_image_data(data_dict, transforms, fallback_transforms=None):
    """Preprocess image data with fallback options for problematic cases"""
    try:
        # Try primary transforms first
        result = transforms(data_dict)
        return result, "primary"
    except Exception as primary_error:
        st.warning(f"Primary preprocessing failed: {str(primary_error)}")
        
        if fallback_transforms is not None:
            try:
                st.info("Attempting fallback preprocessing...")
                result = fallback_transforms(data_dict)
                return result, "fallback"
            except Exception as fallback_error:
                st.error(f"Fallback preprocessing also failed: {str(fallback_error)}")
                raise fallback_error
        else:
            raise primary_error

# ------------------------------
# SESSION STATE INIT
# ------------------------------
if "net" not in st.session_state:
    st.session_state.net = None
if "inferer_cfg" not in st.session_state:
    st.session_state.inferer_cfg = {"roi_size":[128,128,128],"sw_batch_size":1,"overlap":0.25}
if "bundle_path" not in st.session_state:
    st.session_state.bundle_path = ""

# ------------------------------
# SIDEBAR – MODEL SOURCES
# ------------------------------
with st.sidebar:
    st.header("Model Source")
    model_source = st.radio("Choose how to load your model:", [
        "MONAI Bundle from Hugging Face",
        "MONAI Bundle from Local Folder", 
        "Torch/TorchScript File",
        "Fallback Demo UNet",
    ], key="model_source_radio")

    if model_source == "MONAI Bundle from Hugging Face":
        repo_id = st.text_input("Hugging Face repo id (e.g., Project-MONAI/xxx)", key="hf_repo_input")
        if st.button("Download & Load Bundle", key="download_hf_btn") and repo_id:
            with st.spinner("Downloading bundle..."):
                try:
                    net, inferer_cfg, bundle_path = load_bundle_from_hf(repo_id)
                    st.session_state.net = net
                    st.session_state.inferer_cfg = inferer_cfg
                    st.session_state.bundle_path = bundle_path
                    st.success(f"Loaded bundle from {repo_id}")
                    st.caption(f"Local cache: {bundle_path}")
                except Exception as e:
                    st.error(f"Failed to load bundle: {str(e)}")

    elif model_source == "MONAI Bundle from Local Folder":
        folder = st.text_input("Path to local bundle folder", key="local_folder_input")
        if st.button("Load Local Bundle", key="load_local_btn") and folder:
            try:
                net, inferer_cfg = load_bundle_from_local(folder)
                st.session_state.net = net
                st.session_state.inferer_cfg = inferer_cfg
                st.session_state.bundle_path = folder
                st.success("Loaded local bundle")
            except Exception as e:
                st.error(f"Failed to load local bundle: {str(e)}")

    elif model_source == "Torch/TorchScript File":
        up_model = st.file_uploader("Upload .pt/.pth Torch model", type=["pt","pth"], key="torch_uploader")
        arch = st.selectbox("Architecture for state_dict", ["UNet 3D (default)", "UNet 2D (default)"], key="arch_select")
        if st.button("Load Torch model", key="load_torch_btn") and up_model is not None:
            mpath = save_uploaded_file(up_model)
            try:
                net = torch.jit.load(mpath, map_location="cpu")
                st.session_state.net = net
                st.success("TorchScript model loaded")
            except Exception:
                try:
                    is_3d = arch.startswith("UNet 3D")
                    net = make_default_network(is_3d=is_3d)
                    state = torch.load(mpath, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        net.load_state_dict(state["state_dict"])
                    else:
                        net.load_state_dict(state)
                    st.session_state.net = net
                    st.success("State dict model loaded")
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")

    else:  # Demo UNet
        dims = st.selectbox("Demo network dims", ["3D","2D"], key="demo_dims_select")
        in_ch = st.number_input("Input channels",1,4,1, key="in_ch_input")
        out_ch = st.number_input("Output channels/classes",2,10,2, key="out_ch_input")
        if st.button("Create Demo Net", key="create_demo_btn"):
            net = make_default_network(in_channels=in_ch, out_channels=out_ch, is_3d=(dims=="3D"))
            st.session_state.net = net
            st.success("Demo network created (random weights)")

# ------------------------------
# UPLOAD IMAGE
# ------------------------------
st.header("1) Upload Image")
file_type = st.radio("Upload type:", ["NIfTI (.nii/.nii.gz)", "DICOM (ZIP of series)"], key="upload_type_radio")

img_path = ""
img_folder = ""
dicom_files = []

if file_type.startswith("NIfTI"):
    up = st.file_uploader("Upload NIfTI file", type=["nii","nii.gz"], key="nifti_uploader")
    if up is not None:
        try:
            img_path = save_uploaded_file(up)
            st.success(f"Image uploaded: {up.name}")
            st.info(f"File size: {os.path.getsize(img_path)} bytes")
            
            # Test if file can be loaded and show dimensions
            try:
                test_img = nib.load(img_path)
                st.info(f"Image shape: {test_img.shape}, dtype: {test_img.get_fdata().dtype}")
                
                # Warn about potential dimension issues
                if len(test_img.shape) > 3:
                    st.warning(f"Image has {len(test_img.shape)} dimensions. This may cause orientation transform issues. Consider using fallback preprocessing if errors occur.")
                    
            except Exception as load_test_error:
                st.error(f"File validation failed: {load_test_error}")
                img_path = ""
                
        except Exception as save_error:
            st.error(f"Failed to save uploaded file: {save_error}")
            img_path = ""

else:  # DICOM
    up = st.file_uploader("Upload DICOM ZIP", type=["zip"], key="dicom_uploader")
    if up is not None:
        img_folder = extract_zip_to_temp(up.read())
        dicom_files = sorted(glob.glob(os.path.join(img_folder, "**", "*.dcm"), recursive=True) +
                        glob.glob(os.path.join(img_folder, "**", "*.DCM"), recursive=True))
    
    if dicom_files:
        st.success(f"DICOM ZIP extracted ({len(dicom_files)} files found)")
    else:
        st.warning("No DICOM files found in the ZIP. Make sure files are inside ZIP and have .dcm extension.")

# ------------------------------
# PREPROCESSING OPTIONS
# ------------------------------
st.header("2) Preprocessing")
is_3d = st.checkbox("Treat as 3D volume", value=True, key="is_3d_checkbox")
spacing = st.text_input("Target spacing (comma-separated)", value="1.5,1.5,2.0" if is_3d else "0.8,0.8", key="spacing_input")
intens = st.text_input("Intensity range (a_min,a_max)", value="-1000,1000", key="intensity_input")

# Add option for robust preprocessing
robust_preprocessing = st.checkbox("Use robust preprocessing (recommended for problematic images)", value=True, key="robust_preprocessing")

try:
    tgt_spacing, intensity_range = validate_and_parse_inputs(spacing, intens, is_3d)
    a_min, a_max = intensity_range
except ValueError as e:
    st.error(f"Input error: {e}")
    st.stop()

is_dicom = file_type.startswith("DICOM")

# Create both primary and fallback transforms
val_transforms = build_transforms(
    is_3d=is_3d, 
    target_spacing=tgt_spacing, 
    intensity_range=(a_min, a_max),
    is_dicom=is_dicom
)

if robust_preprocessing:
    fallback_transforms = build_safe_transforms(
        is_3d=is_3d,
        target_spacing=tgt_spacing,
        intensity_range=(a_min, a_max),
        is_dicom=is_dicom
    )
else:
    fallback_transforms = None

# ------------------------------
# RUN INFERENCE
# ------------------------------
st.header("3) Run Inference")
net = st.session_state.net
inferer_cfg = st.session_state.inferer_cfg

if net is not None:
    with st.expander("Model Information", expanded=False):
        expected_channels = None
        expected_classes = None
        
        if hasattr(net, 'in_channels'):
            expected_channels = net.in_channels
        if hasattr(net, 'out_channels'):
            expected_classes = net.out_channels
            
        if expected_channels is None:
            for module in net.modules():
                if hasattr(module, 'in_channels'):
                    expected_channels = module.in_channels
                    break
                    
        if expected_classes is None:
            for module in list(net.modules())[-10:]:
                if hasattr(module, 'out_channels'):
                    expected_classes = module.out_channels
                    
        st.write(f"**Expected input channels:** {expected_channels if expected_channels else 'Unknown'}")
        st.write(f"**Expected output classes:** {expected_classes if expected_classes else 'Unknown'}")
        st.write(f"**Model type:** {type(net).__name__}")

if net is None:
    st.warning("Please load or create a model first from the sidebar.")
elif not (img_path or dicom_files):
    st.warning("Please upload an image first.")
else:
    if st.button("Run Inference", key="run_inference_btn"):
        st.info("Running preprocessing and inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)

        try:
            # Prepare data dict(s)
            data_dicts = []
            if img_path:
                data_dicts.append({"image": img_path})
            elif dicom_files:
                data_dicts.append({"image": dicom_files})
            
            # Apply transforms with robust error handling
            transformed = []
            for d in data_dicts:
                try:
                    st.info(f"Applying transforms to: {d.get('image', 'unknown')}")
                    
                    # Use the robust preprocessing function
                    transformed_sample, preprocess_type = preprocess_image_data(
                        d, val_transforms, fallback_transforms
                    )
                    
                    if preprocess_type == "fallback":
                        st.info("✓ Fallback preprocessing successful")
                    else:
                        st.info("✓ Primary preprocessing successful")
                    
                    # Debug: show transformed image info
                    img_tensor = transformed_sample["image"]
                    st.info(f"Transformed image shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
                    
                    transformed.append(transformed_sample)
                    
                except Exception as transform_error:
                    st.error(f"All preprocessing methods failed for {d.get('image', 'unknown')}")
                    
                    # Enhanced error reporting
                    error_str = str(transform_error)
                    if "axcodes must match data_array spatially" in error_str:
                        st.error("❌ Orientation Transform Error: Image dimensions incompatible with orientation transform")
                        st.info("💡 **Solutions:**\n"
                               "- Enable 'Use robust preprocessing'\n"
                               "- Try uploading a different image format\n" 
                               "- Check if image has extra dimensions (4D+)")
                    elif "No such file or directory" in error_str:
                        st.error("File not found - upload may have failed")
                    elif "cannot identify image file" in error_str:
                        st.error("Invalid image format - ensure file is a valid NIfTI or DICOM")
                    elif "Header is not compatible" in error_str:
                        st.error("Corrupted or invalid NIfTI header")
                    
                    with st.expander("Full Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.stop()
            
            if not transformed:
                st.error("No valid data after transformations")
                st.stop()

            # Run inference on transformed data
            for sample in transformed:
                try:
                    image_tensor = sample["image"].unsqueeze(0).to(device)
                    
                    # Handle channel mismatch
                    input_channels = image_tensor.shape[1]
                    expected_channels = None
                    
                    if hasattr(net, 'in_channels'):
                        expected_channels = net.in_channels
                    else:
                        for module in net.modules():
                            if hasattr(module, 'in_channels'):
                                expected_channels = module.in_channels
                                break
                    
                    if expected_channels and input_channels != expected_channels:
                        st.warning(f"Channel mismatch: Model expects {expected_channels} channels, but input has {input_channels}")
                        
                        if input_channels == 1 and expected_channels > 1:
                            st.info(f"Replicating single channel {expected_channels} times to match model input")
                            image_tensor = image_tensor.repeat(1, expected_channels, 1, 1, 1)
                        elif input_channels > 1 and expected_channels == 1:
                            st.info("Using only first channel from multi-channel input")
                            image_tensor = image_tensor[:, :1, ...]
                        elif input_channels > expected_channels:
                            st.info(f"Using first {expected_channels} channels from {input_channels}-channel input")
                            image_tensor = image_tensor[:, :expected_channels, ...]
                        else:
                            st.info(f"Padding input from {input_channels} to {expected_channels} channels with zeros")
                            padding_channels = expected_channels - input_channels
                            padding = torch.zeros(1, padding_channels, *image_tensor.shape[2:], device=device)
                            image_tensor = torch.cat([image_tensor, padding], dim=1)
                    
                    with torch.no_grad():
                        # Configure sliding window inference
                        image_spatial_dims = len(image_tensor.shape) - 2
                        
                        if isinstance(inferer_cfg, dict):
                            roi_size = inferer_cfg.get("roi_size", [128, 128, 128])
                            sw_batch_size = inferer_cfg.get("sw_batch_size", 1)
                            overlap = inferer_cfg.get("overlap", 0.25)
                        else:
                            roi_size = getattr(inferer_cfg, 'roi_size', [128, 128, 128])
                            sw_batch_size = getattr(inferer_cfg, 'sw_batch_size', 1)
                            overlap = getattr(inferer_cfg, 'overlap', 0.25)
                        
                        # Adjust roi_size to match image dimensions
                        if image_spatial_dims == 2 and len(roi_size) == 3:
                            roi_size = roi_size[:2]
                        elif image_spatial_dims == 3 and len(roi_size) == 2:
                            roi_size = roi_size + [128]
                        elif image_spatial_dims != len(roi_size):
                            roi_size = [128] * image_spatial_dims
                        
                        logits = sliding_window_inference(
                            image_tensor,
                            roi_size=roi_size,
                            sw_batch_size=sw_batch_size,
                            predictor=net,
                            overlap=overlap
                        )
                    
                    # Postprocess
                    if logits.shape[1] > 1:
                        pred = torch.argmax(logits, dim=1).squeeze(0)
                    else:
                        pred = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0)

                    # Extract volume data properly
                    vol = image_tensor.squeeze(0)
                    if vol.shape[0] == 1:
                        vol = vol.squeeze(0)
                    vol = vol.cpu().numpy()
                    
                    msk = pred.cpu().numpy().astype(np.uint8)

                    visualize_slices(vol, mask=msk, title="Segmentation")

                    # Save results
                    try:
                        out_path = os.path.join(tempfile.gettempdir(), "prediction_mask.nii.gz")
                        nib.save(nib.Nifti1Image(msk, affine=np.eye(4)), out_path)
                        with open(out_path, "rb") as f:
                            st.download_button( 
                                "Download predicted mask (.nii.gz)", 
                                data=f.read(), 
                                file_name="prediction_mask.nii.gz",
                                mime="application/gzip",
                                key="download_btn"
                            )
                    except Exception as export_error:
                        st.warning(f"Could not export NIfTI: {export_error}")

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        st.error("GPU out of memory - try smaller ROI size or 2D model")
                    else:
                        st.error(f"Inference failed: {str(e)}")
                    st.stop()

            st.success("✅ Inference completed successfully!")

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
            st.stop()

# ------------------------------
# TIPS
# ------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("Tips")
    st.write(
        "- For Hugging Face bundles: pass repo id (e.g., `Project-MONAI/your-bundle`).\n"
        "- For local bundles: point to folder with `configs/` + `models/`.\n"
        "- Adjust spacing & intensity to match modality.\n"
        "- Enable robust preprocessing for 4D+ images.\n"
        "- Sliding window inference used for 3D volumes."
    )