import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="Assistant de Rapport Radiologique AI",
    page_icon="📝",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .findings-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📝 Rapport Radiologique Assisté par GPT-5</h1>', unsafe_allow_html=True)
st.caption("Générer des rapports assistés par IA à partir des observations d'imagerie et du contexte clinique")

# API configuration
API_BASE_URL = st.sidebar.text_input(
    "URL Base de l'API", 
    value="http://localhost:8080",
    help="URL du backend FastAPI"
)

# Main tabs
tab1, tab2, tab3 = st.tabs(["📋 Générer le Rapport", "📊 Téléverser & Traiter", "📚 Bibliothèque de Rapports"])

with tab1:
    st.header("Générer un rapport assisté par IA")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Données des Observations")
        
        # Pre-defined findings templates
        findings_template = st.selectbox(
            "Charger un modèle d'observations:",
            ["Sélectionner un modèle", "Nodules Pulmonaire", "Lésions Hépatique", "IRM Cérébrale", "Personnalisé"]
        )
        
        if findings_template == "Nodules Pulmonaire":
            default_findings = {
                "study_id": "CT-2024-001",
                "modality": "CT",
                "model_version": "lung-nodule-v2.1",
                "findings": {
                    "lesion_count": 3,
                    "largest_lesion_mm": 21.5,
                    "total_volume_mm3": 4500,
                    "locations": ["right_upper_lobe", "left_lower_lobe"],
                    "confidence_scores": {
                        "detection": 0.92,
                        "segmentation": 0.88
                    },
                    "measurements": [
                        {
                            "id": "lesion_1",
                            "location": "right_upper_lobe",
                            "longest_diameter_mm": 21.5,
                            "shortest_diameter_mm": 15.2,
                            "volume_mm3": 2100,
                            "characteristics": {
                                "density_hu": 35.2,
                                "texture": "heterogeneous"
                            }
                        }
                    ]
                }
            }
        elif findings_template == "Lésions Hépatique":
            default_findings = {
                "study_id": "MRI-2024-002",
                "modality": "MRI",
                "model_version": "liver-lesion-v1.5",
                "findings": {
                    "lesion_count": 2,
                    "largest_lesion_mm": 32.1,
                    "locations": ["segment_VI", "segment_VII"],
                    "characteristics": "T2 hyperintense, T1 hypointense"
                }
            }
        elif findings_template == "IRM Cérébrale":
            default_findings = {
                "study_id": "MRI-BRAIN-2024-003",
                "modality": "MRI",
                "model_version": "brain-mri-v3.0",
                "findings": {
                    "abnormalities_detected": True,
                    "white_matter_lesions": 5,
                    "largest_lesion_mm": 8.2,
                    "locations": ["periventricular", "subcortical"]
                }
            }
        else:
            default_findings = {
                "study_id": "STUDY-001",
                "modality": "CT",
                "model_version": "v1.0",
                "findings": {
                    "lesion_count": 1,
                    "largest_lesion_mm": 15.0,
                    "locations": ["organ_region"],
                    "measurements": []
                }
            }
        
        findings_json = st.text_area(
            "JSON des Observations",
            value=json.dumps(default_findings, indent=2),
            height=300,
            help="Observations structurées provenant de l'inférence IA"
        )
    
    with col2:
        st.subheader("Contexte Clinique")
        
        clinical_template = st.selectbox(
            "Modèle de contexte clinique:",
            ["Sélectionner un modèle", "Écranage de Cancer Pulmonaire", "Trouvaille Incidental", "Suivi", "Personnalisé"]
        )
        
        if clinical_template == "Écranage de Cancer Pulmonaire":
            default_context = {
                "age": 65,
                "smoking_history": "30 pack-years, quit 5 years ago",
                "indication": "Écranage de cancer pulmonaire",
                "symptoms": "None",
                "prior_studies": "CT 1 year ago - stable nodules"
            }
        elif clinical_template == "Trouvaille Incidental":
            default_context = {
                "age": 52,
                "indication": "CT douleur abdominale",
                "symptoms": "Right upper quadrant pain",
                "relevant_history": "Hypertension, hyperlipidemia",
                "incidental_finding": True
            }
        elif clinical_template == "Suivi":
            default_context = {
                "age": 48,
                "indication": "Suivi lésion hépatique connue",
                "prior_studies": "MRI 6 months ago - 2.8 cm lesion",
                "comparison": "Stable in size and characteristics"
            }
        else:
            default_context = {
                "age": 50,
                "indication": "Écranage de routine",
                "symptoms": "None reported",
                "relevant_history": "None significant"
            }
        
        clinical_json = st.text_area(
            "JSON du Contexte Clinique",
            value=json.dumps(default_context, indent=2),
            height=200,
            help="Contexte clinique optionnel pour la génération du rapport"
        )
        
        priority = st.selectbox(
            "Priorité",
            ["normal", "urgent"],
            help="Priorité de génération du rapport"
        )
    
    # Generate report button
    if st.button("🚀 Générer le rapport", type="primary", use_container_width=True):
        try:
            # Parse JSON inputs
            findings_data = json.loads(findings_json)
            clinical_data = json.loads(clinical_json) if clinical_json.strip() else {}
            
            # Prepare request
            request_data = {
                "study_id": findings_data.get("study_id", "unknown"),
                "findings": findings_data,
                "clinical_context": clinical_data,
                "priority": priority
            }
            
            # Call API
            with st.spinner("Génération du rapport assisté par IA..."):
                response = requests.post(
                    f"{API_BASE_URL}/report/generate",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    report_data = response.json()
                    
                    st.success("✅ Rapport généré avec succès!")
                    
                    # Display results in expandable sections
                    with st.expander("📋 Rapport structuré", expanded=True):
                        st.json(report_data.get("structured_report", {}))
                    
                    with st.expander("📝 Narratif radiologique", expanded=True):
                        st.text_area(
                            "Narratif",
                            value=report_data.get("radiology_narrative", ""),
                            height=200,
                            label_visibility="collapsed"
                        )
                    
                    with st.expander("👥 Résumé du patient", expanded=True):
                        st.text_area(
                            "Résumé du patient",
                            value=report_data.get("patient_summary", ""),
                            height=150,
                            label_visibility="collapsed"
                        )
                    
                    # Report metadata
                    st.info(f"ID du rapport : {report_data.get('report_id')} | Généré à : {report_data.get('generated_at')}")
                    
                else:
                    st.error(f"❌ Erreur API : {response.status_code} - {response.text}")
                    
        except json.JSONDecodeError:
            st.error("❌ Format JSON invalide dans les observations ou le contexte clinique.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur de connexion : {str(e)}")
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")

with tab2:
    st.header("Téléverser & Traiter des Images")
    
    uploaded_file = st.file_uploader(
        "Téléverser une image médicale",
        type=["nii", "nii.gz", "dcm", "zip"],
        help="Téléverser un fichier NIfTI, DICOM ou archive ZIP DICOM"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            modality = st.selectbox(
                "Modalité",
                ["CT", "MRI", "X-Ray", "Ultrasound", "PET"],
                index=0
            )
            
            model_version = st.text_input(
                "Version du Modèle",
                value="v1.0",
                help="Version du modèle d'IA à utiliser"
            )
            
            if st.button("🔬 Traiter l'image", use_container_width=True):
                try:
                    # Upload and process
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {
                        "modality": modality,
                        "model_version": model_version
                    }
                    
                    with st.spinner("Traitement de l'image avec MONAI AI..."):
                        response = requests.post(
                            f"{API_BASE_URL}/infer/study-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.current_job = result
                            st.success(f"✅ {result.get('message')}")
                            st.info(f"ID de job : {result.get('job_id')}")
                        else:
                            st.error(f"❌ Traitement échoué : {response.text}")
                            
                except Exception as e:
                    st.error(f"❌ Erreur de téléversement : {str(e)}")
        
        with col2:
            if "current_job" in st.session_state:
                job = st.session_state.current_job
                
                if st.button("🔄 Vérifier l'état", use_container_width=True):
                    try:
                        status_response = requests.get(
                            f"{API_BASE_URL}/infer/status/{job.get('job_id').replace('job_', '')}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data.get("status") == "completed":
                                st.success("✅ Traitement terminé!")
                                
                                # Get results
                                results_response = requests.get(
                                    f"{API_BASE_URL}/infer/results/{job.get('job_id').replace('job_', '')}",
                                    timeout=10
                                )
                                
                                if results_response.status_code == 200:
                                    results = results_response.json()
                                    st.session_state.current_results = results
                                    st.json(results)
                            else:
                                st.info(f"⏳ Statut : {status_data.get('status')}")
                        else:
                            st.error("❌ Échec de la vérification de l'état")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur de vérification d'état : {str(e)}")

with tab3:
    st.header("Bibliothèque de Rapports")
    
    try:
        # Fetch reports from API
        response = requests.get(f"{API_BASE_URL}/reports", timeout=10)
        
        if response.status_code == 200:
            reports_data = response.json()
            reports = reports_data.get("reports", [])
            
            if reports:
                st.write(f"Nombre de rapports : {len(reports)}")
                
                for report in reports:
                    report_data = report.get("response", {})
                    request_data = report.get("request", {})
                    
                    with st.expander(f"Rapport {report_data.get('report_id')} - {request_data.get('study_id')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Observations structurées**")
                            st.json(request_data.get('findings', {}))
                        
                        with col2:
                            st.write("**Rapport généré**")
                            st.text_area(
                                "Narratif radiologique",
                                value=report_data.get('radiology_narrative', ''),
                                height=150,
                                label_visibility="collapsed"
                            )
                            
                            if st.button("📋 Voir le rapport complet", key=report_data.get('report_id')):
                                st.json(report_data)
            else:
                st.info("Aucun rapport trouvé dans la bibliothèque")
        else:
            st.error("Échec de la récupération des rapports depuis l'API")
            
    except Exception as e:
        st.error(f"Erreur lors de l'accès à la bibliothèque de rapports : {str(e)}")

# Footer
st.markdown("---")
st.caption("""
**Assistant de Rapport Radiologique AI** - Cet outil aide les radiologues en générant des rapports structurés à partir des observations d'IA. Tous les rapports doivent être revus et vérifiés par des professionnels médico‑qualifiés avant utilisation clinique.
""")