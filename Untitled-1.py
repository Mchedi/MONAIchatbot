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
    EnsureTyped, ResizeD
)
from monai.inferers import sliding_window_inference
from monai.bundle import ConfigParser
from monai.networks.nets import UNet
from monai.transforms import Lambdad

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(page_title="MONAI 1.5 Application Web", layout="wide")
st.title("🩺 MONAI 1.5 – Application d'Inférence d'Imagerie Médicale")
st.caption("Téléchargez une image, choisissez un modèle, lancez l'inférence, et visualisez les coupes.")

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

def build_transforms(is_3d: bool,
                     target_spacing: Tuple[float, ...],
                     intensity_range: Tuple[float, float],
                     is_dicom: bool = False,
                     keys=("image",)):
    # 1️⃣ Grab the inferer_cfg object – it may be a dict or a component
    inferer_cfg_obj = getattr(st.session_state, "inferer_cfg", None)

    # 2️⃣ Create a dict‑like view that always supports .get()
    if isinstance(inferer_cfg_obj, dict):
        inferer_cfg_dict = inferer_cfg_obj
    else:
        # Try to expose the component’s attributes as a dict
        try:
            inferer_cfg_dict = vars(inferer_cfg_obj)
        except Exception:
            inferer_cfg_dict = {}   # fallback empty dict

    # 3️⃣ Now we can safely use .get()
    roi = inferer_cfg_dict.get("roi_size")

    # -----------------------------------------------------------------
    # Convert to a tuple of ints and keep the right dimensionality
    # -----------------------------------------------------------------
    if roi is not None:
        max_dims = 3 if is_3d else 2
        roi = tuple(int(v) for v in roi[:max_dims])

    tr = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=keys, func=lambda x: x.squeeze()),
        EnsureChannelFirstd(keys=keys),
    ]

    # OPTIONAL: force the expected spatial size (highly recommended)
    if roi is not None:
        tr.append(ResizeD(keys=keys, spatial_size=roi, mode="area"))

    if not is_dicom:
        try:
            tr.append(Orientationd(keys=keys, axcodes="RAS"))
        except Exception:
            st.warning("Ignorer la transformation d'orientation en raison de dimensions incompatibles")

    tr.extend([
        Spacingd(keys=keys, pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=keys,
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
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
    st.info(f"Forme du volume avant traitement : {original_shape}")
    
    # Handle multi-channel volumes (4D: channels, depth, height, width)
    if volume.ndim == 4:
        if volume.shape[0] == 1:
            # Single channel case
            volume = volume.squeeze(0)
            st.info(f"Volume à canal unique réduit à la forme : {volume.shape}")
        else:
            # Multi-channel case - take first channel or create composite
            st.warning(f"Volume multi-canaux détecté ({volume.shape[0]} canaux). Utilisation du premier canal pour la visualisation.")
            volume = volume[0]  # Take first channel
            st.info(f"Première canal utilisé, nouvelle forme : {volume.shape}")
    
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
        st.error(f"Impossible de visualiser le volume avec {volume.ndim} dimensions")
        return

    # Create visualization
    cols = st.columns(len(slices))
    for col, (img, msk, name) in zip(cols, slices):
        try:
            fig, ax = plt.subplots(figsize=(4,4))
            
            # Ensure image is 2D for matplotlib
            if img.ndim > 2:
                st.warning(f"L'image de coupe est encore {img.ndim}D, on prend la première tranche/canal")
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
            col.error(f"Échec de la visualisation pour {name} : {str(viz_error)}")
            st.info(f"Forme de l'image : {img.shape}, Forme du masque : {msk.shape if msk is not None else 'None'}")

def validate_and_parse_inputs(spacing_input: str, intensity_input: str, is_3d: bool):
    """Validate and parse spacing and intensity inputs"""
    spacing_values = [x.strip() for x in spacing_input.split(",")]
    expected_dims = 3 if is_3d else 2
    if len(spacing_values) != expected_dims:
        raise ValueError(f"L'espacement doit contenir {expected_dims} valeurs pour {'3D' if is_3d else '2D'}")
    
    intensity_values = [x.strip() for x in intensity_input.split(",")]
    if len(intensity_values) != 2:
        raise ValueError("L'intervalle d'intensité doit contenir exactement 2 valeurs (a_min,a_max)")
    
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
        st.warning(f"Le prétraitement principal a échoué : {str(primary_error)}")
        
        if fallback_transforms is not None:
            try:
                st.info("Tentative de prétraitement de secours...")
                result = fallback_transforms(data_dict)
                return result, "fallback"
            except Exception as fallback_error:
                st.error(f"Le prétraitement de secours a également échoué : {str(fallback_error)}")
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
    st.header("Source du Modèle")
    model_source = st.radio(
        "Choisissez comment charger votre modèle :",
        [
            "Lot MONAI depuis Hugging Face",
            "Lot MONAI depuis un Dossier Local",
            "Fichier Torch/TorchScript",
            "UNet de démonstration (fallback)",
        ],
        key="model_source_radio")

    if model_source == "Lot MONAI depuis Hugging Face":
        repo_id = st.text_input("Identifiant du repo Hugging Face (ex.: Project-MONAI/xxx)", key="hf_repo_input")
        if st.button("Télécharger & Charger le Lot", key="download_hf_btn") and repo_id:
            with st.spinner("Téléchargement du lot..."):
                try:
                    net, inferer_cfg, bundle_path = load_bundle_from_hf(repo_id)
                    st.session_state.net = net
                    st.session_state.inferer_cfg = inferer_cfg
                    st.session_state.bundle_path = bundle_path
                    st.success(f"Lot chargé depuis {repo_id}")
                    st.caption(f"Cache local : {bundle_path}")
                except Exception as e:
                    st.error(f"Échec du chargement du lot : {str(e)}")

    elif model_source == "Lot MONAI depuis un Dossier Local":
        folder = st.text_input("Chemin du dossier du bundle local", key="local_folder_input")
        if st.button("Charger le Bundle Local", key="load_local_btn") and folder:
            try:
                net, inferer_cfg = load_bundle_from_local(folder)
                st.session_state.net = net
                st.session_state.inferer_cfg = inferer_cfg
                st.session_state.bundle_path = folder
                st.success("Lot local chargé")
            except Exception as e:
                st.error(f"Échec du chargement du lot local : {str(e)}")

    elif model_source == "Fichier Torch/TorchScript":
        up_model = st.file_uploader("Uploader le modèle Torch (.pt/.pth)", type=["pt","pth"], key="torch_uploader")
        arch = st.selectbox("Architecture du state_dict", ["UNet 3D (default)", "UNet 2D (default)"], key="arch_select")
        if st.button("Charger le modèle Torch", key="load_torch_btn") and up_model is not None:
            mpath = save_uploaded_file(up_model)
            try:
                net = torch.jit.load(mpath, map_location="cpu")
                st.session_state.net = net
                st.success("Modèle TorchScript chargé")
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
                    st.success("Modèle state_dict chargé")
                except Exception as e:
                    st.error(f"Échec du chargement du modèle : {str(e)}")

    else:  # Demo UNet
        dims = st.selectbox("Dimensions du réseau de démonstration", ["3D","2D"], key="demo_dims_select")
        in_ch = st.number_input("Canaux d'entrée",1,4,1, key="in_ch_input")
        out_ch = st.number_input("Canaux de sortie/classes",2,10,2, key="out_ch_input")
        if st.button("Créer le réseau de démonstration", key="create_demo_btn"):
            net = make_default_network(in_channels=in_ch, out_channels=out_ch, is_3d=(dims=="3D"))
            st.session_state.net = net
            st.success("Réseau de démonstration créé (poids aléatoires)")

# ------------------------------
# UPLOAD IMAGE
# ------------------------------
st.header("1) Importer une image")
file_type = st.radio("Type de téléchargement :", ["NIfTI (.nii/.nii.gz)", "DICOM (ZIP de série)"], key="upload_type_radio")

img_path = ""
img_folder = ""
dicom_files = []

if file_type.startswith("NIfTI"):
    up = st.file_uploader("Uploader un fichier NIfTI", type=["nii","nii.gz"], key="nifti_uploader")
    if up is not None:
        try:
            img_path = save_uploaded_file(up)
            st.success(f"Image téléversée : {up.name}")
            st.info(f"Taille du fichier : {os.path.getsize(img_path)} octets")
            
            # Test if file can be loaded and show dimensions
            try:
                test_img = nib.load(img_path)
                st.info(f"Forme de l'image : {test_img.shape}, type de données : {test_img.get_fdata().dtype}")
                
                # Warn about potential dimension issues
                if len(test_img.shape) > 3:
                    st.warning(f"L'image possède {len(test_img.shape)} dimensions. Cela peut causer des problèmes avec la transformation d'orientation. Envisagez d'utiliser le prétraitement de secours si des erreurs surviennent.")
                    
            except Exception as load_test_error:
                st.error(f"Échec de la validation du fichier : {load_test_error}")
                img_path = ""
                
        except Exception as save_error:
            st.error(f"Échec de l'enregistrement du fichier téléversé : {save_error}")
            img_path = ""

else:  # DICOM
    up = st.file_uploader("Uploader le ZIP DICOM", type=["zip"], key="dicom_uploader")
    if up is not None:
        img_folder = extract_zip_to_temp(up.read())
        dicom_files = sorted(
            glob.glob(os.path.join(img_folder, "**", "*.dcm"), recursive=True) +
            glob.glob(os.path.join(img_folder, "**", "*.DCM"), recursive=True)
        )
    
    if dicom_files:
        st.success(f"ZIP DICOM extrait ({len(dicom_files)} fichiers trouvés)")
    else:
        st.warning("Aucun fichier DICOM trouvé dans le ZIP. Assurez-vous que les fichiers sont à l'intérieur du ZIP et ont l'extension .dcm.")

# ------------------------------
# PREPROCESSING OPTIONS
# ------------------------------
st.header("2) Prétraitement")
is_3d = st.checkbox("Traiter comme un volume 3D", value=True, key="is_3d_checkbox")
spacing = st.text_input("Espacement cible (séparé par des virgules)", value="1.5,1.5,2.0" if is_3d else "0.8,0.8", key="spacing_input")
intens = st.text_input("Intervalle d'intensité (a_min,a_max)", value="-1000,1000", key="intensity_input")

# Add option for robust preprocessing
robust_preprocessing = st.checkbox("Utiliser le prétraitement robuste (recommandé pour les images problématiques)", value=True, key="robust_preprocessing")

try:
    tgt_spacing, intensity_range = validate_and_parse_inputs(spacing, intens, is_3d)
    a_min, a_max = intensity_range
except ValueError as e:
    st.error(f"Erreur d'entrée : {e}")
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
st.header("3) Lancer l'Inférence")
net = st.session_state.net
inferer_cfg = st.session_state.inferer_cfg

if net is not None:
    with st.expander("Informations sur le Modèle", expanded=False):
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
                    
        st.write(f"**Canaux d'entrée attendus :** {expected_channels if expected_channels else 'Inconnu'}")
        st.write(f"**Classes de sortie attendues :** {expected_classes if expected_classes else 'Inconnu'}")
        st.write(f"**Type de modèle :** {type(net).__name__}")

if net is None:
    st.warning("Veuillez d'abord charger ou créer un modèle depuis la barre latérale.")
elif not (img_path or dicom_files):
    st.warning("Veuillez d'abord télécharger une image.")
else:
    if st.button("Lancer l'Inférence", key="run_inference_btn"):
        st.info("Exécution du prétraitement et de l'inférence...")
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
                    st.info(f"Application des transformations à : {d.get('image', 'unknown')}")
                    
                    # Use the robust preprocessing function
                    transformed_sample, preprocess_type = preprocess_image_data(
                        d, val_transforms, fallback_transforms
                    )
                    
                    if preprocess_type == "fallback":
                        st.info("✓ Prétraitement de secours réussi")
                    else:
                        st.info("✓ Prétraitement principal réussi")
                    
                    # Debug: show transformed image info
                    img_tensor = transformed_sample["image"]
                    st.info(f"Forme de l'image transformée : {img_tensor.shape}, type : {img_tensor.dtype}")
                    
                    transformed.append(transformed_sample)
                    
                except Exception as transform_error:
                    st.error(f"Toutes les méthodes de prétraitement ont échoué pour {d.get('image', 'unknown')}")
                    
                    # Enhanced error reporting
                    error_str = str(transform_error)
                    if "axcodes must match data_array spatially" in error_str or "Orientationd" in error_str:
                        st.error("❌ Erreur de transformation d'orientation : dimensions de l'image incompatibles avec la transformation d'orientation")
                        st.info("💡 **Solutions :**\n"
                                "- Activer le prétraitement robuste\n"
                                "- Essayer de téléverser un format d'image différent\n"
                                "- Vérifier si l'image possède des dimensions supplémentaires (4D+)")
                    elif "No such file or directory" in error_str:
                        st.error("Fichier introuvable - le téléversement a peut‑être échoué")
                    elif "cannot identify image file" in error_str:
                        st.error("Format d'image invalide - assurez‑vous que le fichier est un NIfTI ou DICOM valide")
                    elif "Header is not compatible" in error_str:
                        st.error("En‑tête NIfTI corrompu ou invalide")
                    
                    with st.expander("Détails complets de l'erreur"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.stop()
            
            if not transformed:
                st.error("Aucune donnée valide après les transformations")
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
                        st.warning(f"Mauvaise correspondance de canaux : le modèle attend {expected_channels} canaux, mais l'entrée en possède {input_channels}")
                        
                        if input_channels == 1 and expected_channels > 1:
                            st.info(f"Réplication du canal unique {expected_channels} fois pour correspondre à l'entrée du modèle")
                            image_tensor = image_tensor.repeat(1, expected_channels, 1, 1, 1)
                        elif input_channels > 1 and expected_channels == 1:
                            st.info("Utilisation du premier canal uniquement")
                            image_tensor = image_tensor[:, :1, ...]
                        elif input_channels > expected_channels:
                            st.info(f"Utilisation des {expected_channels} premiers canaux d'une entrée à {input_channels} canaux")
                            image_tensor = image_tensor[:, :expected_channels, ...]
                        else:
                            st.info(f"Remplissage de l'entrée de {input_channels} à {expected_channels} canaux avec des zéros")
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
                                "Télécharger le masque prédit (.nii.gz)", 
                                data=f.read(), 
                                file_name="prediction_mask.nii.gz",
                                mime="application/gzip",
                                key="download_btn"
                            )
                    except Exception as export_error:
                        st.warning(f"Impossible d'exporter le fichier NIfTI : {export_error}")

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        st.error("Mémoire GPU insuffisante - essayez une taille de ROI plus petite ou un modèle 2D")
                    else:
                        st.error(f"Échec de l'inférence : {str(e)}")
                    st.stop()

            st.success("✅ Inférence terminée avec succès !")

        except Exception as e:
            st.error(f"Erreur inattendue : {str(e)}")
            import traceback
            st.error(f"Trace complète : {traceback.format_exc()}")
            st.stop()

# ------------------------------
# TIPS
# ------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("Conseils")
    st.write(
        "- Pour les bundles Hugging Face : indiquez l'ID du repo (ex.: `Project-MONAI/votre-bundle`).\n"
        "- Pour les bundles locaux : indiquez le dossier contenant `configs/` + `models/`.\n"
        "- Ajustez l'espacement et l'intensité en fonction de la modalité.\n"
        "- Activez le prétraitement robuste pour les images 4D+.\n"
        "- L'inférence en fenêtre glissante est utilisée pour les volumes 3D."
    )